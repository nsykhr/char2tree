{
  "dataset_reader": {
    "type": "ud_char_level"
  },
  "train_data_path": "data/UD_Japanese-GSD/ja_gsd-ud-train.conllu",
  "validation_data_path": "data/UD_Japanese-GSD/ja_gsd-ud-dev.conllu",
  "vocabulary": {
    "non_padded_namespaces": ["*tags", "*labels", "upos", "dependency"]
  },
  "model": {
    "type": "char_level_joint",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 100,
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
    "batch_size": 64,
    "biggest_batch_first": true,
    "sorting_keys": [["chars", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 4e-3,
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
    "grad_norm": 5.0,
    "cuda_device": 0
  }
}