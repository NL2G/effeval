import torch

torch.set_float32_matmul_precision("medium")
import argparse as ap
import logging

import pandas as pd
from comet import download_model, load_from_checkpoint
from datasets import Dataset
from rich.logging import RichHandler
from tqdm import tqdm

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        prog="pseudo-label.py", description="Pseudo-labeling script"
    )
    parser.add_argument(
        "--model", type=str, help="Model name", default="Unbabel/wmt22-comet-da"
    )
    parser.add_argument("--gpus", type=int, help="Number of GPUs", default=1)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=1024)
    parser.add_argument("--input", type=str, help="Input file", required=True)
    parser.add_argument("--column", type=str, help="Column name", default="score")
    parser.add_argument("--out", type=str, help="Output file", required=True)
    parser.add_argument("--shard", type=int, help="Shard id", default=0)

    args = parser.parse_args()

    logger.info(f"Running pseudo-labeling script with args: {args}")

    # Load model
    logger.info(f"Loading model {args.model}")
    _path = download_model(args.model)
    model = load_from_checkpoint(_path)

    # Load dataset
    logger.info(f"Loading dataset {args.input}")
    dataset = Dataset.load_from_disk(args.input)
    dataset = dataset.shard(num_shards=2, index=args.shard, contiguous=True)
    dataset = dataset.to_dict()
    dataset = [
        {
            "src": src,
            "mt": mt,
            "ref": ref,
        }
        for src, mt, ref in tqdm(
            zip(dataset["src"], dataset["mt"], dataset["ref"]), unit_scale=True
        )
    ]

    # Pseudo-label
    logger.info(f"Predicting pseudo-labels")
    predicts = model.predict(
        dataset, batch_size=args.batch_size, gpus=args.gpus, num_workers=1
    )

    # add pseudo-labels to dataset
    logger.info(f"Adding pseudo-labels to dataset")
    for i, predict in enumerate(predicts.scores):
        dataset[i][args.column] = predict

    # save dataset
    logger.info(f"Saving dataset")
    dataset = pd.DataFrame(dataset)
    dataset.to_csv(args.out, index=False, encoding="utf-8")
