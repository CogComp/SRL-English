{
    "dataset_reader": {
        "type": "preposition_srl",
        "bert_model_name": "bert-base-uncased",
    },
 
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 16
        }
    },
 
    "train_data_path": std.extVar("SRL_TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("SRL_VALIDATION_DATA_PATH"),
    "test_data_path": std.extVar("SRL_TEST_DATA_PATH"),
 
    "model": {
        "type": "preposition_srl_bert",
        "embedding_dropout": 0.1,
        "bert_model": "bert-base-uncased",
    },
 
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],                
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 500,
            "num_steps_per_epoch": 8829,
        },

        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },

        "grad_norm": 1.0,
        "num_epochs": 500,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 0,
    },
}
