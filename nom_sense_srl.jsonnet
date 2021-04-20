{
    "dataset_reader": {
      "type": "nombank-sense-srl",
      "bert_model_name": "bert-base-uncased",
    },

    "iterator": {
	"type": "bucket",
	"batch_size": 32,
	"sorting_keys": [["tokens", "num_tokens"]]
    },

    "train_data_path": "/shared/celinel/noun_srl/read_nombank/preprocess_nombank/train.srl",
    "validation_data_path": "/shared/celinel/noun_srl/read_nombank/preprocess_nombank/development.srl",
    "test_data_path": "/shared/celinel/noun_srl/read_nombank/preprocess_nombank/test.srl",

    "model": {
        "type": "nombank-sense-srl-bert",
        "embedding_dropout": 0.1,
        "bert_model": "bert-base-uncased",
    },

    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 5e-5,
	    "t_total": -1,
	    "max_grad_norm": 1.0,            
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 15,
            "num_steps_per_epoch": 8829,
        },
        "num_serialized_models_to_keep": 2,
        "num_epochs": 15,
        "validation_metric": "+combined-score",
        "cuda_device": [0,1],
	"should_log_learning_rate": true,
    },

}
