from rich.logging import RichHandler
import logging
import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
import transformers as tr
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from accelerate import Accelerator

from data_utils import make_preprocessing_fn, load_from_config
from model.comet import Comet
from config import DATA_CONFIG, TRAINING_CONFIG

import argparse as ap

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def prepare_args():

    parser: ap.ArgumentParser = ap.ArgumentParser(
        description="Trainable metrics for MT evaluation",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        prog="run.py",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="comet",
        help="Data configuration to use",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="comet",
        help="Model configuration to use",
    )
    parser.add_argument(
        "--use-adapters",
        default=True,
        action="store_true",
        help="Use adapters",
    )
    parser.add_argument(
        "--dev-size",
        type=float,
        default=0.01,
        help="Size of the dev set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="Random seed",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log every n steps",
    )
    cli_args: ap.Namespace = parser.parse_args()
    data_args = DATA_CONFIG[cli_args.data_config]
    model_args = TRAINING_CONFIG[cli_args.model_config]

    common_config: ap.Namespace = ap.Namespace(dict(**data_args, **model_args))
    common_config.use_adapters = cli_args.use_adapters
    common_config.dev_size = cli_args.dev_size
    common_config.seed = cli_args.seed
    common_config.log_every = cli_args.log_every
    return common_config

def main(common_config: ap.Namespace):

    accelerator = Accelerator(log_with="wandb", split_batches=True)
    accelerator.init_trackers(project_name='trainable-metrics', config=dict(common_config))

    logger.info(f"Using following arguments: {common_config}")
    tr.set_seed(common_config.seed)

    logger.info("Loading data")
    train, dev, test = load_from_config(
        common_config.train, 
        common_config.test, 
        common_config.dev_size,
        common_config.seed
    )

    logger.info("Loading tokenizer")
    tokenizer = tr.XLMRobertaTokenizerFast.from_pretrained(common_config.encoder_model_name, use_fast=True)

    logger.info("Preparing preprocessing function")
    preprocessing_fn = make_preprocessing_fn(tokenizer, max_length=512)

    logger.info("Preprocessing data")
    columns_to_remove = train.column_names
    train = train.map(preprocessing_fn, batched=True, remove_columns=columns_to_remove, num_proc=4)
    dev = dev.map(preprocessing_fn, batched=True, remove_columns=columns_to_remove, num_proc=4)
    test = test.map(preprocessing_fn, batched=True, remove_columns=columns_to_remove, num_proc=4)

    logger.info("Preparing data loaders")
    data_collator = tr.DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True, 
        pad_to_multiple_of=8, 
        max_length=512, 
        return_tensors="pt"
    )
    train_loader = DataLoader(train, batch_size=common_config.batch_size, shuffle=True, collate_fn=data_collator)
    dev_loader = DataLoader(dev, batch_size=common_config.batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test, batch_size=common_config.batch_size, shuffle=False, collate_fn=data_collator)

    logger.info("Preparing model")
    model = Comet(
        encoder_model_name=common_config.encoder_model_name,
        use_adapters=common_config.use_adapters,
        layer=common_config.layer,
        keep_embeddings_freezed=common_config.keep_embeddings_freezed,
        hidden_sizes=common_config.hidden_sizes,
        activations=common_config.activations,
        final_activation=common_config.final_activation,
        pad_token_id=tokenizer.pad_token_id,
        dropout=common_config.dropout,
    )

    logger.info("Preparing optimizer")
    encoder_params = model.encoder.layerwise_lr(common_config.encoder_lr, common_config.layerwise_decay)
    top_layers_parameters = [
        {"params": model.estimator.parameters(), "lr": common_config.estimator_lr},
    ]
    if model.layerwise_attention:
        layerwise_attn_params = [
            {
                "params": model.layerwise_attention.parameters(),
                "lr": common_config.estimator_lr,
            }
        ]
        params = encoder_params + top_layers_parameters + layerwise_attn_params
    else:
        params = encoder_params + top_layers_parameters

    optimizer = tr.AdamW(params, lr=common_config.encoder_lr)

    accelerator.wait_for_everyone()
    logger.info("Model placement")
    model, optimizer, train_loader, dev_loader = accelerator.prepare(
        model, optimizer, train_loader, dev_loader
    )

    logger.info("Training")
    accelerator.wait_for_everyone()

    dev_kendall_tau = []
    dev_loss = []

    patience_counter: int = 0

    for epoch in range(common_config.max_epochs):
        logger.info(f"Epoch {epoch}")
        model.train()
        if common_config.nr_frozen_epochs > 0:
            logger.info(f"Freezing encoder for {common_config.nr_frozen_epochs} epochs")
            model.encoder.freeze()
            
        for i, batch in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
            optimizer.zero_grad()
            labels = batch.pop("labels")
            preds = model(**batch).squeeze()
            loss = F.mse_loss(preds, labels)
            accelerator.backward(loss)
            optimizer.step()
            loss_item = loss.item()
            
            labels_for_metrics, preds_for_metrics = accelerator.gather_for_metrics((labels, preds))
            train_kendall_tau = kendalltau(
                labels_for_metrics.cpu(),
                preds_for_metrics.cpu()
            ).statistic

            if i % common_config.log_every == 0:
                accelerator.log({
                    "train_loss": loss_item,
                    "train_kendall_tau": train_kendall_tau,
                })

            if i / len(train_loader) > common_config.nr_frozen_epochs:
                logger.info(f"Unfreezing encoder")
                model.encoder.unfreeze()

        model.eval()
        epoch_dev_loss = []
        epoch_dev_kendall_tau = []
        for i, batch in enumerate(tqdm(dev_loader, disable=not accelerator.is_main_process)):
            with torch.no_grad():
                labels = batch.pop("labels")
                preds = model(**batch).squeeze()
                loss = F.mse_loss(preds, labels)
                loss_item = loss.item()
                labels_for_metrics, preds_for_metrics = accelerator.gather_for_metrics((labels, preds))
                epoch_dev_kendall_tau.append(kendalltau(
                    labels_for_metrics.cpu(),
                    preds_for_metrics.cpu()
                ).statistic)
                epoch_dev_loss.append(loss_item)

        dev_kendall_tau_value = np.mean(epoch_dev_kendall_tau)
        dev_loss_value = np.mean(epoch_dev_loss)
        accelerator.log({
            "dev_loss": dev_loss,
            "dev_kendall_tau": dev_kendall_tau,
        })
        logger.info(f"Dev loss: {dev_loss:.4f} | Dev Kendall Tau: {dev_kendall_tau:.4f}")

        dev_kendall_tau.append(dev_kendall_tau_value)
        dev_loss.append(dev_loss_value)

        if dev_kendall_tau_value != max(dev_kendall_tau):
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}")
        else:
            patience_counter = 0

        if patience_counter >= common_config.patience:
            logger.info("Early stopping")
            break

    if accelerator.is_main_process:
        logger.info("Evaluating on test set")
        model.eval()
        test_preds = []
        test_labels = []
        for i, batch in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                preds = model(**batch).squeeze()
                test_preds += preds.cpu().numpy().tolist()
                test_labels += labels.cpu().numpy().tolist()

        test_kendall_tau = kendalltau(test_labels, test_preds).statistic
        logger.info(f"Test Kendall Tau: {test_kendall_tau:.4f}")
        accelerator.log({"test_kendall_tau": test_kendall_tau})

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training finished")
