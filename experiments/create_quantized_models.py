import torch
import transformers as tr
import tempfile
from functools import partial
import numpy as np
from datasets import Dataset
from load_data import data, load_data
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
from rich.logging import RichHandler
from rich.progress import track
import logging
import time

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

models = {
    #"bert-base-uncased": {
    #    "model": "bert-base-uncased",
    #    "n_layers": 9,
    #    "repo_id": "nllg/bert-base-uncased-quantized-9L-{quantization_mode}-{hardware}",
    #},
    #"distilbert-base-uncased": {
    #    "model": "distilbert-base-uncased",
    #    "n_layers": 5,
    #    "repo_id": "nllg/distilbert-base-uncased-quantized-5L-{quantization_mode}-{hardware}",
    #},
    #"tinybert": {
    #    "model": "huawei-noah/TinyBERT_General_4L_312D",
    #    "n_layers": 4,
    #    "repo_id": "nllg/tinybert-quantized-4L-{quantization_mode}-{hardware}",
    #},
    "bert-tiny": {
        "model": "google/bert_uncased_L-2_H-128_A-2",
        "n_layers": 1,
        "repo_id": "Rexhaif/bert-tiny-{quantization_mode}-quantized-1L-{hardware}",
    },
}

def load_wmt_data(max_size: int = 2500) -> Dataset:
    wmt_data = data.copy()
    load_data(wmt_data, split_data=0)
    all_strings = []
    for dataset in ['wmt15', 'wmt16', 'wmt21']:
        all_strings += wmt_data[dataset]['refs']
        all_strings += wmt_data[dataset]['hyps']

    dataset = Dataset.from_dict({
        'text': all_strings,
    })

    dataset = dataset.shuffle(seed=42)

    return dataset.select(range(max_size))

def preprocess_function(tokenizer, examples):
    return tokenizer(examples['text'], truncation=True)

def prepare_model(model_name: str, n_layers: int) -> str:
    model = tr.AutoModel.from_pretrained(model_name)
    tokenizer = tr.AutoTokenizer.from_pretrained(model_name)

    assert (
        0 <= n_layers <= len(model.encoder.layer)
    ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
    model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:n_layers]])  

    model.config.num_hidden_layers = n_layers

    tmp_dir = tempfile.mkdtemp()

    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    return tmp_dir

def create_dynamically_quantized_model(model_path: str, hardware: str) -> str:
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_path, from_transformers=True)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    if hardware == "avx2":
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    elif hardware == "arm64":
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    else:
        raise ValueError(f"Invalid hardware: {hardware}")

    quantizer.quantize(qconfig, model_path)
    return model_path


def create_static_quantized_model(model_path: str, hardware: str, dataset: Dataset) -> ORTModelForFeatureExtraction:
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_path, from_transformers=True)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    if hardware == "avx2":
        qconfig = AutoQuantizationConfig.avx2(is_static=True, per_channel=False)
    elif hardware == "arm64":
        qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    else:
        raise ValueError(f"Invalid hardware: {hardware}")

    tokenizer = tr.AutoTokenizer.from_pretrained(model_path)

    def preprocess_function(tokenizer, examples):
        return tokenizer(examples['text'], truncation=True, return_attention_mask=True, return_token_type_ids=True)

    calibration_dataset = dataset.map(
        partial(preprocess_function, tokenizer),
        batched=True, 
        remove_columns=["text"]
    )

    logger.info(f"Calibrating on dataset {calibration_dataset}")
    calibration_config = AutoCalibrationConfig.minmax(dataset)

    ranges = quantizer.fit(
        calibration_dataset, calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize
    )

    logger.info(f"Quantization ranges: {ranges}")

    quantizer.quantize(
        save_dir=model_path,
        quantization_config=qconfig,
        calibration_tensors_range=ranges,
    )
    return model_path

def create_non_quantized_model(model_path: str) -> ORTModelForFeatureExtraction:
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_path, from_transformers=True)
    ort_model.save_pretrained(model_path, file_name="model.onnx")
    return model_path


def test_model(model_path: str, test_dataset: Dataset, file_name: str) -> None:
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_path, file_name=file_name)
    torch_model = tr.AutoModel.from_pretrained(model_path)
    torch_model.eval()
    tokenizer = tr.AutoTokenizer.from_pretrained(model_path)
    collator = tr.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    test_dataset = test_dataset.map(
        partial(preprocess_function, tokenizer),
        batched=True, 
        remove_columns=["text"]
    )
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=collator)
    differences = {
        'cosine': [],
        'l2': [],
    }
    times = {
        'torch': [],
        'ort': [],
    }
    n_tokens = []
    for batch in track(test_dataloader, description="Testing quantized model"):
        with torch.no_grad():
            n_tokens.append(batch['input_ids'].numel())
            t1 = time.time()
            torch_outputs = torch_model(**batch).last_hidden_state.view(-1).unsqueeze(0)
            t2 = time.time()
            ort_outputs = ort_model(**batch).last_hidden_state.view(-1).unsqueeze(0)
            t3 = time.time()
            differences['cosine'].append(
                torch.nn.functional.cosine_similarity(torch_outputs, ort_outputs).mean().item()
            )
            differences['l2'].append(
                torch.nn.functional.mse_loss(torch_outputs, ort_outputs).mean().item()
            )
            times['torch'].append(t2 - t1)
            times['ort'].append(t3 - t2)

    logger.info(f"Average cosine similarity: {np.mean(differences['cosine']):.4f}")
    logger.info(f"Average L2 distance: {np.mean(differences['l2']):.4f}")
    logger.info(f"Average torch inference time (tokens/s): {np.sum(n_tokens) / np.sum(times['torch']):.4f}")
    logger.info(f"Average ORT inference time (tokens/s): {np.sum(n_tokens) / np.sum(times['ort']):.4f}")
    
if __name__ == "__main__":

    logger.info("Loading WMT data")
    test_dataset = load_wmt_data()

    for model_name, model_info in models.items():
        logger.info(f"Creating quantized models for {model_name} with {model_info['n_layers']} layers")
        for quantization_mode in ["dynamic", "non", "static"]:
            hardware_list = ['avx2', 'arm64'] if quantization_mode != "non" else ['any']
            for hardware in hardware_list:
                model_path = prepare_model(model_info["model"], model_info["n_layers"])
                logger.info(f"Creating {quantization_mode} quantized model for {hardware}")
                if quantization_mode == "dynamic":
                    model_path = create_dynamically_quantized_model(model_path, hardware)
                    file_name = "model_quantized.onnx"
                elif quantization_mode == "static":
                    model_path = create_static_quantized_model(model_path, hardware, test_dataset)
                    file_name = "model_quantized.onnx"
                elif quantization_mode == "non":
                    file_name = "model.onnx"
                    model_path = create_non_quantized_model(model_path)
                else:
                    raise ValueError(f"Invalid quantization_mode: {quantization_mode}")

                logger.info("Testing quantized model")
                test_model(model_path, test_dataset, file_name)

                logger.info("Pushing quantized model to Hugging Face Hub")
                model = ORTModelForFeatureExtraction.from_pretrained(model_path, file_name=file_name)
                repository_id = model_info["repo_id"].format(quantization_mode=quantization_mode, hardware=hardware)
                model.push_to_hub(model_path, repository_id)

    logger.info("Done")
