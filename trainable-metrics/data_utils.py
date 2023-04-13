import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path: str, lps: list[str] | str = "all", domain: str = "news") -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)
    if lps != "all":
        dataset = dataset[dataset["lp"].isin(lps)]
    if domain != "all":
        dataset = dataset[dataset["domain"] == domain]

    scaler = MinMaxScaler()
    dataset = dataset[["src", "mt", "ref", "score"]]
    dataset["score"] = scaler.fit_transform(dataset[["score"]])
    return dataset

def load_from_config(
        train_config: list[dict[str, str | list[str]]],
        test_config: dict[str, str | list[str]], 
        dev_size: float = 0.01, 
        random_seed: int = 999
    ) -> tuple[Dataset, Dataset, Dataset]:

    train_data = []
    for element in train_config:
        train_data.append(load_data(**element))
    train_data = pd.concat(train_data)
    train_dev = Dataset.from_pandas(train_data)
    train_dev = train_dev.train_test_split(test_size=dev_size, seed=random_seed)

    test_data = load_data(**test_config)
    test_data = Dataset.from_pandas(test_data)

    return train_dev["train"], train_dev["test"], test_data

def make_preprocessing_fn(tokenizer, max_length: int = 512):
    def preprocessing_function(examples):
        src_inputs = tokenizer(
            examples["src"],
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        mt_inputs = tokenizer(
            examples["mt"],
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        ref_inputs = tokenizer(
            examples["ref"],
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        result = {
            "src_input_ids": src_inputs.input_ids,
            "src_attention_mask": src_inputs.attention_mask,
            "mt_input_ids": mt_inputs.input_ids,
            "mt_attention_mask": mt_inputs.attention_mask,
            "ref_input_ids": ref_inputs.input_ids,
            "ref_attention_mask": ref_inputs.attention_mask,
        }
        if "score" in examples:
            result["labels"] = examples["score"]
        return result
    return preprocessing_function
