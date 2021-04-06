{
  "dataset_reader": {
    "type": "ud_reader",
    "character_level": false
  },
  "train_data_path": "data/UD_Russian-Syntagrus/ru_syntagrus-ud-train.conllu",
  "validation_data_path": "data/UD_Russian-Syntagrus/ru_syntagrus-ud-dev.conllu",
  "vocabulary": {
    "non_padded_namespaces": ["*tags", "*labels", "upos", "dependency"]
  },
  "model": {
    "type": "joint_tagger_parser",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 300,
      "hidden_size": 640,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.3,
      "layer_dropout_probability": 0.3,
      "use_highway": true
    },
    "upos_hidden": 320,
    "embedding_dropout": 0.3,
    "encoded_dropout": 0.3,
    "upos_dropout": 0.3,
    "mlp_dropout": 0.3
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 512,
    "biggest_batch_first": true,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
      "weight_decay": 1e-4
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "patience": 5,
      "min_lr": 1e-5
    },
    "num_epochs": 100,
    "patience": 15,
    "grad_norm": 3.0,
    "cuda_device": 0
  }
}