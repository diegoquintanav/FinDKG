# Training scrip for KGTransformer model
# --------------------------------------
#
# Train cml:
#   python train_DKG_run.py
#

import argparse
from enum import Enum
import os
import pprint
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dkg
from dkg.eval import evaluate
from dkg.model import (
    DKG_DEFAULT_CONFIG,
    Combiner,
    DynamicGraphModel,
    EdgeModel,
    EmbeddingUpdater,
    InterEventTimeModel,
    MultiAspectEmbedding,
)
from dkg.model.time_interval_transform import TimeIntervalTransform
from dkg.train import compute_loss, pack_checkpoint, unpack_checkpoint
from dkg.utils.log_utils import add_logger_file_handler, get_log_root_path, logger
from dkg.utils.model_utils import get_embedding
from dkg.utils.train_utils import (
    EarlyStopping,
)


INTER_EVENT_TIME_DTYPE = torch.float32


############################### Config ###############################


class KGTransformerDataset(str, Enum):
    FinDKG = "FinDKG"
    ICEWS18 = "ICEWS18"
    ICEWS14 = "ICEWS14"
    ICEWS_500 = "ICEWS_500"
    GDELT = "GDELT"
    WIKI = "WIKI"
    YAGO = "YAGO"


class KGTransformerModelVersion(str, Enum):
    KGTransformer = "KGTransformer"
    GraphTransformer = "GraphTransformer"


class KGTransformerModelType(str, Enum):
    KGT_RNN = "KGT+RNN"  # for GraphTransformer
    RGCN_RNN = "RGCN+RNN"  # for GraphRNN


class EnumAction(argparse.Action):
    """Adds Enum support to argparse.

    See https://dnmtechs.com/enhancing-argparse-with-support-for-enum-arguments-in-python-3/
    """

    def __init__(self, **kwargs):
        choices = kwargs.pop("choices")
        super(EnumAction, self).__init__(**kwargs)
        self.choices = [choice.value for choice in choices]

    def __call__(self, parser, namespace, values, option_string=None):
        value = values[0]
        for choice in self.choices:
            if value == choice:
                setattr(namespace, self.dest, choice)
                return
        parser.error(f"Invalid choice: {value}. Available choices are {self.choices}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train KGTransformer model",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=KGTransformerDataset,
        action=EnumAction,
        default=KGTransformerDataset.FinDKG.value,
        choices=KGTransformerDataset,
        help="Specify the dataset",
    )
    parser.add_argument(
        "--model-version",
        type=KGTransformerModelVersion,
        default=KGTransformerModelVersion.KGTransformer.value,
        action=EnumAction,
        choices=KGTransformerModelVersion,
        help="Model name",
    )
    parser.add_argument(
        "--model-type",
        type=KGTransformerModelType,
        default=KGTransformerModelType.KGT_RNN.value,
        action=EnumAction,
        choices=KGTransformerModelType,
        help="Model type",
    )
    parser.add_argument("--epoch-times", type=int, default=150, help="Number of epochs")
    parser.add_argument("--random-seed", type=int, default=41, help="Random seed")
    parser.add_argument(
        "--data-root-path", type=str, default="./FinDKG_dataset", help="Data root path"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model", dest="flag_train"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the model", dest="flag_eval"
    )

    return parser.parse_args()


def cli(args: argparse.Namespace):
    graph_mode = args.dataset.value
    model_ver = args.model_version.value
    model_type = args.model_type.value
    epoch_times = args.epoch_times
    random_seed = args.random_seed
    data_root_path = args.data_root_path
    flag_train = args.flag_train
    flag_eval = args.flag_eval

    print(dkg.__version__)

    ############################### Load Graph Data ###############################
    G = dkg.data.load_temporal_knowledge_graph(graph_mode, data_root=data_root_path)
    collate_fn = partial(dkg.utils.collate_fn, G=G)

    train_data_loader = DataLoader(G.train_times, shuffle=False, collate_fn=collate_fn)
    val_data_loader = DataLoader(G.val_times, shuffle=False, collate_fn=collate_fn)
    test_data_loader = DataLoader(G.test_times, shuffle=False, collate_fn=collate_fn)

    ############################### Model Config ###############################
    cfg = deepcopy(DKG_DEFAULT_CONFIG)

    cfg.seed = random_seed  # Random Seed
    cfg.cuda = cfg.gpu >= 0 and torch.cuda.is_available()
    cfg.device = torch.device("cuda:{}".format(cfg.gpu) if cfg.cuda else "cpu")
    cfg.graph = graph_mode
    cfg.version = model_ver  #'GTransformer'
    cfg.optimize = "both"

    dim_num = 200  # default dim set up to 200
    cfg.static_entity_embed_dim = dim_num
    cfg.structural_dynamic_entity_embed_dim = dim_num
    cfg.temporal_dynamic_entity_embed_dim = dim_num

    cfg.num_gconv_layers = 2  # layer of the KGTransformer, default set up 2

    cfg.num_attn_heads = 8
    cfg.lr = 0.0005  # leanrning rate

    # Freeze the random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    num_relations = G.num_relations

    # Training dir
    log_root_path = get_log_root_path(cfg.graph, cfg.log_dir)

    # mkdir if not exist
    os.makedirs(log_root_path, exist_ok=True)

    print(log_root_path)
    overall_best_checkpoint_prefix = (
        f"{cfg.graph}_{cfg.version}_overall_best_checkpoint"
    )
    run_best_checkpoint_prefix = f"{cfg.graph}_{cfg.version}_run_best_checkpoint"

    with open(
        os.path.join(log_root_path, f"{cfg.graph}_args_{cfg.version}.txt"),
        "w",
    ) as f:
        f.write(pprint.pformat(cfg.__dict__))

    add_logger_file_handler(cfg.graph, cfg.version, ".", log_time=None, fname_prefix="")

    logger.info(f"  --> Set up random seed [ {cfg.seed} ]")

    ######################### Build up KG #########################

    node_latest_event_time = torch.zeros(
        G.number_of_nodes(), G.number_of_nodes() + 1, 2, dtype=INTER_EVENT_TIME_DTYPE
    )
    time_interval_transform = TimeIntervalTransform(log_transform=True)

    # KG embedding module
    embedding_updater = EmbeddingUpdater(
        G.number_of_nodes(),
        cfg.static_entity_embed_dim,
        cfg.structural_dynamic_entity_embed_dim,
        cfg.temporal_dynamic_entity_embed_dim,
        node_latest_event_time,
        G.num_relations,
        cfg.rel_embed_dim,
        num_node_types=G.num_node_types,
        num_heads=cfg.num_attn_heads,  # number of attention head for Transformer
        graph_structural_conv=model_type,
        graph_temporal_conv=model_type,
        num_gconv_layers=cfg.num_gconv_layers,
        num_rnn_layers=cfg.num_rnn_layers,
        time_interval_transform=time_interval_transform,
        dropout=cfg.dropout,
        activation=cfg.embedding_updater_activation,
        graph_name=cfg.graph,
    ).to(cfg.device)

    combiner = Combiner(
        cfg.static_entity_embed_dim,
        cfg.structural_dynamic_entity_embed_dim,
        cfg.static_dynamic_combine_mode,
        cfg.combiner_gconv,
        G.num_relations,
        cfg.dropout,
        cfg.combiner_activation,
    ).to(cfg.device)

    edge_model = EdgeModel(
        G.number_of_nodes(),
        G.num_relations,
        cfg.rel_embed_dim,
        combiner,
        dropout=cfg.dropout,
    ).to(cfg.device)

    inter_event_time_model = InterEventTimeModel(
        dynamic_entity_embed_dim=cfg.temporal_dynamic_entity_embed_dim,
        static_entity_embed_dim=cfg.static_entity_embed_dim,
        num_rels=G.num_relations,
        rel_embed_dim=cfg.rel_embed_dim,
        num_mix_components=cfg.num_mix_components,
        time_interval_transform=time_interval_transform,
        inter_event_time_mode=cfg.inter_event_time_mode,
        dropout=cfg.dropout,
    )

    model = DynamicGraphModel(
        embedding_updater,
        combiner,
        edge_model,
        inter_event_time_model,
        node_latest_event_time,
    ).to(cfg.device)

    # Init the static and dynamic entity & relation embeddings
    static_entity_embeds = MultiAspectEmbedding(
        structural=get_embedding(
            G.num_nodes(), cfg.static_entity_embed_dim, zero_init=False
        ),
        temporal=get_embedding(
            G.num_nodes(), cfg.static_entity_embed_dim, zero_init=False
        ),
    )

    init_dynamic_entity_embeds = MultiAspectEmbedding(
        structural=get_embedding(
            G.num_nodes(),
            [cfg.num_rnn_layers, cfg.structural_dynamic_entity_embed_dim],
            zero_init=True,
        ),
        temporal=get_embedding(
            G.num_nodes(),
            [cfg.num_rnn_layers, cfg.temporal_dynamic_entity_embed_dim, 2],
            zero_init=True,
        ),
    )

    init_dynamic_relation_embeds = MultiAspectEmbedding(
        structural=get_embedding(
            G.num_relations,
            [cfg.num_rnn_layers, cfg.rel_embed_dim, 2],
            zero_init=True,
        ),
        temporal=get_embedding(
            G.num_relations,
            [cfg.num_rnn_layers, cfg.rel_embed_dim, 2],
            zero_init=True,
        ),
    )
    ####################################################################################################

    ############################## Graph Model Training ##############################

    # > ES lerning system
    stopper = EarlyStopping(
        cfg.graph,
        cfg.patience,
        result_root=log_root_path,
        run_best_checkpoint_prefix=run_best_checkpoint_prefix,
        overall_best_checkpoint_prefix=overall_best_checkpoint_prefix,
        eval=cfg.eval,
    )

    params = list(model.parameters()) + [
        static_entity_embeds.structural,
        static_entity_embeds.temporal,
        init_dynamic_entity_embeds.structural,
        init_dynamic_entity_embeds.temporal,
        init_dynamic_relation_embeds.structural,
        init_dynamic_relation_embeds.temporal,
    ]
    edge_optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    time_optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    ######################## > Training secetion ##############################
    if flag_train:
        # Prestore the model
        model.eval()
        model.node_latest_event_time.zero_()
        node_latest_event_time.zero_()
        dynamic_entity_emb = init_dynamic_entity_embeds
        dynamic_relation_emb = init_dynamic_relation_embeds

        val_dict, val_dynamic_entity_emb, val_dynamic_relation_emb, loss_weights = (
            evaluate(
                model,
                val_data_loader,
                G,
                static_entity_embeds,
                dynamic_entity_emb,
                dynamic_relation_emb,
                num_relations,
                cfg,
                "Validation",
                cfg.full_link_pred_validation,
                cfg.time_pred_eval,
                0,
            )
        )

        # Save the model locally
        node_latest_event_time_post_valid = deepcopy(model.node_latest_event_time)
        pack_args = (
            model,
            edge_optimizer,
            time_optimizer,
            static_entity_embeds,
            init_dynamic_entity_embeds,
            val_dynamic_entity_emb,
            init_dynamic_relation_embeds,
            val_dynamic_relation_emb,
            node_latest_event_time_post_valid,
            loss_weights,
            cfg,
        )

        score = val_dict["MRR"]
        stopper.step(score, pack_checkpoint(*pack_args))

        # Start training along epochs
        for epoch in range(epoch_times):
            ######### Training ##########
            model.train()
            epoch_start_time = time.time()

            dynamic_entity_emb_post_train, dynamic_relation_emb_post_train = None, None

            model.node_latest_event_time.zero_()
            node_latest_event_time.zero_()
            dynamic_entity_emb = init_dynamic_entity_embeds
            dynamic_relation_emb = init_dynamic_relation_embeds

            num_train_batches = len(train_data_loader)
            train_tqdm = tqdm(train_data_loader)

            epoch_train_loss_dict = defaultdict(list)
            batch_train_loss = 0
            batches_train_loss_dict = defaultdict(list)

            for batch_i, (prior_G, batch_G, cumul_G, _batch_times) in enumerate(
                train_tqdm
            ):
                train_tqdm.set_description(
                    f"[Training / epoch-{epoch} / batch-{batch_i}]"
                )
                last_batch = batch_i == num_train_batches - 1

                # Based on the current entity embeddings, predict edges in batch_G and compute training loss
                batch_train_loss_dict = compute_loss(
                    model,
                    cfg.optimize,
                    batch_G,
                    static_entity_embeds,
                    dynamic_entity_emb,
                    dynamic_relation_emb,
                    cfg,
                )
                batch_train_loss += sum(batch_train_loss_dict.values())

                for loss_term, loss_val in batch_train_loss_dict.items():
                    epoch_train_loss_dict[loss_term].append(loss_val.item())
                    batches_train_loss_dict[loss_term].append(loss_val.item())

                if batch_i > 0 and (
                    (batch_i % cfg.rnn_truncate_every == 0) or last_batch
                ):
                    # noinspection PyUnresolvedReferences
                    batch_train_loss.backward()
                    batch_train_loss = 0

                    if cfg.optimize in ["edge", "both"]:
                        edge_optimizer.step()
                        edge_optimizer.zero_grad()
                    if cfg.optimize in ["time", "both"]:
                        time_optimizer.step()
                        time_optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    if (
                        cfg.embedding_updater_structural_gconv
                        or cfg.embedding_updater_temporal_gconv
                    ):
                        for emb in dynamic_entity_emb + dynamic_relation_emb:
                            emb.detach_()

                    tqdm.write(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')} [Epoch {epoch:03d}-Batch {batch_i:03d}] "
                        f"batch train loss total={sum([sum(loss) for loss in batches_train_loss_dict.values()]):.4f} | "
                        f"{', '.join([f'{loss_term}={sum(loss_cumul):.4f}' for loss_term, loss_cumul in batches_train_loss_dict.items()])}"
                    )
                    batches_train_loss_dict = defaultdict(list)

                dynamic_entity_emb, dynamic_relation_emb = (
                    model.embedding_updater.forward(
                        prior_G,
                        batch_G,
                        cumul_G,
                        static_entity_embeds,
                        dynamic_entity_emb,
                        dynamic_relation_emb,
                        cfg.device,
                    )
                )

                if last_batch:
                    dynamic_entity_emb_post_train = dynamic_entity_emb
                    dynamic_relation_emb_post_train = dynamic_relation_emb

            epoch_end_time = time.time()
            logger.info(
                f"[Epoch-{epoch}] Train loss total={sum([sum(loss) for loss in epoch_train_loss_dict.values()]):.4f} | "
                f"{', '.join([f'{loss_term}={sum(loss_cumul):.4f}' for loss_term, loss_cumul in epoch_train_loss_dict.items()])} | "
                f"elapsed time={epoch_end_time - epoch_start_time:.4f} secs"
            )

            ######### Validation #########
            if epoch >= cfg.eval_from and epoch % cfg.eval_every == 0:
                dynamic_entity_emb = dynamic_entity_emb_post_train
                dynamic_relation_emb = dynamic_relation_emb_post_train

                (
                    val_dict,
                    val_dynamic_entity_emb,
                    val_dynamic_relation_emb,
                    loss_weights,
                ) = evaluate(
                    model,
                    val_data_loader,
                    G,
                    static_entity_embeds,
                    dynamic_entity_emb,
                    dynamic_relation_emb,
                    num_relations,
                    cfg,
                    "Validation",
                    cfg.full_link_pred_validation,
                    cfg.time_pred_eval,
                    epoch,
                )

                node_latest_event_time_post_valid = deepcopy(
                    model.node_latest_event_time
                )

                if cfg.early_stop:
                    criterion = cfg.early_stop_criterion
                    if cfg.early_stop_criterion not in val_dict:
                        criterion = "loss"

                    if criterion == "MRR":
                        score = val_dict[criterion]
                    else:  # MAE, loss
                        score = -val_dict[criterion]
                    pack_args = (
                        model,
                        edge_optimizer,
                        time_optimizer,
                        static_entity_embeds,
                        init_dynamic_entity_embeds,
                        val_dynamic_entity_emb,
                        init_dynamic_relation_embeds,
                        val_dynamic_relation_emb,
                        node_latest_event_time_post_valid,
                        loss_weights,
                        cfg,
                    )
                    if stopper.step(score, pack_checkpoint(*pack_args)):
                        logger.info(f"[Epoch-{epoch}] Early stop!")
                        break

    #################### > Test Set Evaluation ################################
    if flag_eval:
        run_best_checkpoint = stopper.load_checkpoint()

        (
            model,
            edge_optimizer,
            time_optimizer,
            val_static_entity_emb,
            init_dynamic_entity_embeds,
            val_dynamic_entity_emb,
            _init_dynamic_relation_embeds,
            val_dynamic_relation_emb,
            node_latest_event_time_post_valid,
            loss_weights,
            _,
        ) = unpack_checkpoint(
            run_best_checkpoint,
            model,
            edge_optimizer,
            time_optimizer,
        )

        logger.info("Loaded the best model so far for testing.")

        model.node_latest_event_time.copy_(node_latest_event_time_post_valid)

        test_start_time = time.time()
        evaluate(
            model,
            test_data_loader,
            G,
            val_static_entity_emb,
            val_dynamic_entity_emb,
            val_dynamic_relation_emb,
            num_relations,
            cfg,
            "Test",  # specify the test set
            full_link_pred_eval=cfg.full_link_pred_test,
            time_pred_eval=cfg.time_pred_eval,
            loss_weights=loss_weights,
        )
        test_end_time = time.time()
        logger.info(f"Test elapsed time={test_end_time - test_start_time:.4f} secs")


if __name__ == "__main__":
    cfg = parse_args()
    cli(cfg)
