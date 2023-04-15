from rich.logging import RichHandler
import logging
import torch
import time
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers as tr
import numpy as np
from scipy.stats import kendalltau
from accelerate import Accelerator

from data_utils import make_preprocessing_fn, load_from_config, make_collate_fn
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
        default=False,
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
    parser.add_argument(
        "--no-tqdm",
        default=False,
        action="store_true",
        help="Disable tqdm",
    )
    parser.add_argument(
        "--adapter-config",
        type=str,
        default="pfeiffer",
        help="Adapter configuration to use",
    )
    cli_args: ap.Namespace = parser.parse_args()
    data_args = DATA_CONFIG[cli_args.data_config]
    model_args = TRAINING_CONFIG[cli_args.model_config]

    common_config: ap.Namespace = ap.Namespace(**dict(**data_args, **model_args))
    common_config.use_adapters = cli_args.use_adapters
    common_config.dev_size = cli_args.dev_size
    common_config.seed = cli_args.seed
    common_config.log_every = cli_args.log_every
    common_config.no_tqdm = cli_args.no_tqdm
    common_config.adapter_config = cli_args.adapter_config
    return common_config


def get_n_tokens(batch):
    
    total = 0
    
    with torch.no_grad():
        for segment in {'src', 'mt', 'ref'}:
            attn = batch[f"{segment}_attention_mask"].cpu().detach()
            n_tokens = torch.sum(torch.flatten(attn)).item()
            total += n_tokens
            
    return total


def main(common_config: ap.Namespace):

    accelerator = Accelerator(log_with="wandb", split_batches=True)
    accelerator.init_trackers(project_name='trainable-metrics', config=vars(common_config))

    logger.info(f"Using following arguments: {common_config}")
    tr.enable_full_determinism(common_config.seed)

    logger.info("Loading data")
    train, dev, test = load_from_config(
        common_config.train, 
        common_config.test, 
        common_config.dev_size,
        common_config.seed
    )

    logger.info("Loading tokenizer")
    tokenizer = tr.XLMRobertaTokenizer.from_pretrained(common_config.encoder_model_name)

    logger.info("Preparing preprocessing function")
    preprocessing_fn = make_preprocessing_fn(tokenizer, max_length=512)

    logger.info("Preprocessing data")
    columns_to_remove = train.column_names
    train = train.map(preprocessing_fn, batched=True, remove_columns=columns_to_remove, num_proc=16)
    dev = dev.map(preprocessing_fn, batched=True, remove_columns=columns_to_remove, num_proc=8)
    test = test.map(preprocessing_fn, batched=True, remove_columns=columns_to_remove, num_proc=8)

    logger.info("Preparing data loaders")
    data_collator = make_collate_fn(tokenizer, max_length=512)
    train_loader = DataLoader(train, batch_size=common_config.batch_size, shuffle=True, collate_fn=data_collator, num_workers=2)
    dev_loader = DataLoader(dev, batch_size=1, shuffle=False, collate_fn=data_collator, num_workers=2)
    test_loader = DataLoader(test, batch_size=16, shuffle=False, collate_fn=data_collator, num_workers=2)

    logger.info("Preparing model")
    model = Comet(
        encoder_model_name=common_config.encoder_model_name,
        use_adapters=common_config.use_adapters,
        adapter_config=common_config.adapter_config,
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

    optimizer = torch.optim.AdamW(params, lr=common_config.encoder_lr)

    accelerator.wait_for_everyone()
    logger.info("Model placement")
    model, optimizer, train_loader, dev_loader = accelerator.prepare(
        model, optimizer, train_loader, dev_loader
    )

    logger.info("Training")
    accelerator.wait_for_everyone()

    dev_kendall_tau = []
    dev_loss = []
    
    model_forward_time = []
    model_backward_time = []
    model_n_tokens = []
    model_memory_usage = []

    patience_counter: int = 0
    is_frozen: bool = False

    tqdm_disable: bool = not accelerator.is_main_process or common_config.no_tqdm

    if common_config.nr_frozen_epochs > 0:
        logger.info(f"Freezing encoder for {common_config.nr_frozen_epochs} epochs")
        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)
        model.encoder.freeze()
        is_frozen = True
        accelerator.clear()
        model = accelerator.prepare(model)
        accelerator.wait_for_everyone()

    for epoch in range(common_config.max_epochs):
        logger.info(f"Epoch {epoch}")
        model.train()
        
        logger.info(f"Training Epoch #{epoch}")
        for i, batch in enumerate(tqdm(train_loader, disable=tqdm_disable)):

            if i % common_config.log_every == 0:
                torch.cuda.reset_peak_memory_stats()

            optimizer.zero_grad()
            labels = batch.pop("labels")
            t_0 = time.perf_counter()
            preds = model(**batch).squeeze()
            loss = F.mse_loss(preds, labels)
            t_1 = time.perf_counter()
            accelerator.backward(loss)
            t_2 = time.perf_counter()
            optimizer.step()

            if i % common_config.log_every == 0:

                peak_memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

                peak_memory_usage = peak_memory_usage / 1024 / 1024
                model_memory_usage.append(peak_memory_usage)

                n_tokens = get_n_tokens(batch)
                model_n_tokens.append(n_tokens)
                model_forward_time.append(t_1 - t_0)
                model_backward_time.append(t_2 - t_1)
                
                backward_per_tokens = n_tokens / (t_2 - t_1)
                forward_per_tokens = n_tokens / (t_1 - t_0)
                
                labels_for_metrics, preds_for_metrics = accelerator.gather_for_metrics((labels, preds))
                train_kendall_tau = kendalltau(
                    labels_for_metrics.cpu().detach(),
                    preds_for_metrics.cpu().detach()
                ).statistic

                loss_item = loss.item()
                accelerator.log({
                    "train_loss": loss_item,
                    "train_kendall_tau": train_kendall_tau,
                    'n_tokens': n_tokens,
                    'gpu_mem_usage_mb': peak_memory_usage,
                    'forward_speed_tokens_per_second': forward_per_tokens,
                    'backward_speed_tokens_per_second': backward_per_tokens,
                })
                if common_config.no_tqdm:
                    logger.info(f"Step: {i:^5} => Train loss: {loss_item:^5} | Train Kendall Tau: {train_kendall_tau:^5} | GPU mem usage: {peak_memory_usage:^5}MB | Forward speed: {forward_per_tokens:^5} tok/s | Backward speed: {backward_per_tokens:^5} tok/s")

            if is_frozen and (i / len(train_loader) > common_config.nr_frozen_epochs):
                logger.info(f"Unfreezing encoder")
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model.encoder.unfreeze()
                model.encoder.freeze_embeddings()
                is_frozen = False
                accelerator.clear()
                model = accelerator.prepare(model)
                accelerator.wait_for_everyone()

        model.eval()
        epoch_dev_loss = []
        epoch_preds = []
        epoch_labels = []
        logger.info(f"Evaluating Epoch #{epoch}")
        for i, batch in enumerate(tqdm(dev_loader, disable=tqdm_disable)):
            with torch.no_grad():
                labels = batch.pop("labels")
                preds = model(**batch).squeeze()
                loss = F.mse_loss(preds, labels)
                loss_item = loss.item()
                labels_for_metrics, preds_for_metrics = accelerator.gather_for_metrics((labels, preds))
                epoch_preds.append(preds_for_metrics.item())
                epoch_labels.append(labels_for_metrics.item())
                epoch_dev_loss.append(loss_item)

        dev_kendall_tau_value = kendalltau(epoch_labels, epoch_preds).statistic
        dev_loss_value = np.mean(epoch_dev_loss)
        accelerator.log({
            "dev_loss": dev_loss_value,
            "dev_kendall_tau": dev_kendall_tau_value,
        })
        logger.info(f"Dev loss: {dev_loss_value:.4f} | Dev Kendall Tau: {dev_kendall_tau_value:.4f}")

        dev_kendall_tau.append(dev_kendall_tau_value)
        dev_loss.append(dev_loss_value)

        if dev_kendall_tau_value != max(dev_kendall_tau):
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}")
        else:
            logger.info(f"Kenall Tau improved, resetting patience counter")
            patience_counter = 0

        if patience_counter >= common_config.patience:
            logger.info("Early stopping")
            break

    if accelerator.is_main_process:
        logger.info("Evaluating on test set")
        model.eval()
        test_preds = []
        test_labels = []
        for i, batch in enumerate(tqdm(test_loader, disable=tqdm_disable)):
            with torch.no_grad():
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                preds = model(**batch).squeeze()
                test_preds += preds.cpu().detach().tolist()
                test_labels += labels.cpu().detach().tolist()

        test_kendall_tau = kendalltau(test_labels, test_preds).statistic
        logger.info(f"Test Kendall Tau: {test_kendall_tau:.4f}")
        accelerator.log({"test_kendall_tau": test_kendall_tau})

    accelerator.wait_for_everyone()

    mean_forward_time = np.mean(model_forward_time)
    mean_backward_time = np.mean(model_backward_time)
    std_forward_time = np.std(model_forward_time)
    std_backward_time = np.std(model_backward_time)
    median_forward_time = np.median(model_forward_time)
    median_backward_time = np.median(model_backward_time)

    forward_speed = np.array(model_n_tokens) / np.array(model_forward_time)
    backward_speed = np.array(model_n_tokens) / np.array(model_backward_time)
    mean_forward_speed = np.mean(forward_speed)
    mean_backward_speed = np.mean(backward_speed)
    std_forward_speed = np.std(forward_speed)
    std_backward_speed = np.std(backward_speed)
    median_forward_speed = np.median(forward_speed)
    median_backward_speed = np.median(backward_speed)

    mean_memory_usage = np.mean(model_memory_usage)
    std_memory_usage = np.std(model_memory_usage)
    median_memory_usage = np.median(model_memory_usage)

    logger.info(f"Forward time:   {mean_forward_time:.4f} +- {std_forward_time:.4f} (median: {median_forward_time:.4f})")
    logger.info(f"Backward time:  {mean_backward_time:.4f} +- {std_backward_time:.4f} (median: {median_backward_time:.4f})")
    logger.info(f"Forward speed:  {mean_forward_speed:.4f} +- {std_forward_speed:.4f}  (median: {median_forward_speed:.4f})")
    logger.info(f"Backward speed: {mean_backward_speed:.4f} +- {std_backward_speed:.4f} (median: {median_backward_speed:.4f})")
    logger.info(f"Memory usage:   {mean_memory_usage:.4f} +- {std_memory_usage:.4f} (median: {median_memory_usage:.4f})")

    accelerator.log({
        "mean_forward_time": mean_forward_time,
        "mean_backward_time": mean_backward_time,
        "std_forward_time": std_forward_time,
        "std_backward_time": std_backward_time,
        "mean_forward_speed": mean_forward_speed,
        "mean_backward_speed": mean_backward_speed,
        "std_forward_speed": std_forward_speed,
        "std_backward_speed": std_backward_speed,
        "mean_memory_usage": mean_memory_usage,
        "std_memory_usage": std_memory_usage,
        "median_forward_time": median_forward_time,
        "median_backward_time": median_backward_time,
        "median_forward_speed": median_forward_speed,
        "median_backward_speed": median_backward_speed,
        "median_memory_usage": median_memory_usage,
    })

    accelerator.end_training()
    logger.info("Training finished")

    

if __name__ == "__main__":
    args = prepare_args()
    main(args)