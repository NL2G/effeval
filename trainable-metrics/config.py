DATA_CONFIG = {
    "comet": {
        "train": [
            #{
            #    "path": "./data/2017-da.tar.gz",
            #    "lps": "all",
            #},
            #{
            #    "path": "./data/2018-da.tar.gz",
            #    "lps": "all",
            #},
            #{
            #    "path": "./data/2019-da.tar.gz",
            #    "lps": "all",
            #},
            {
                "path": "./data/2020-da.tar.gz",
                "lps": ["en-de", "en-zh", "en-ru", "ru-en", "de-en", "zh-en"],
            }
        ],
        "test": {
            "path": "./data/2022-da.tar.gz",
            "lps": [
                "zh-en",
                "ru-en",
                "de-en",
            ],
        }
    },
    "cometinho": {
        "train": [
            {
                "path": "",
                "lps": "all",
            }
        ],
        "test": {
            "path": "./data/2022-da.tar.gz",
            "lps": [
                "zh-en",
                "ru-en",
                "de-en",
            ],
        }
    }
}

TRAINING_CONFIG = {
    "comet": {
        "nr_frozen_epochs": 0.01,
        'keep_embeddings_freezed': True,
        "encoder_lr": 1.0e-6,
        "estimator_lr": 1.5e-5,
        "layerwise_decay": 0.95,
        "encoder_model_name": "xlm-roberta-large",
        "layer": "mix",
        "batch_size": 16,
        "hidden_sizes": [3072, 1024],
        "activations": "Tanh",
        "final_activation": None,
        "layer_transformation": "sparsemax",
        "max_epochs": 4,
        "patience": 2,
        "dropout": 0.1,
    },
    "cometinho": {
        
    }
}