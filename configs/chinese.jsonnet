{
  "dataset_reader": {
    "type": "ud_reader",
    "use_xpos": true
  },
  "train_data_path": "data/UD_Chinese-GSD/zh_gsd-ud-train.conllu",
  "validation_data_path": "data/UD_Chinese-GSD/zh_gsd-ud-dev.conllu",
  "vocabulary": {
    "non_padded_namespaces": ["*tags", "*labels", "upos", "xpos", "dependency"]
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
      "hidden_size": 384,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.33,
      "layer_dropout_probability": 0.33,
      "use_highway": true
    },
    "upos_hidden": 192,
    "xpos_head": true,
    "xpos_hidden": 192,
    "arc_mlp_size": 256,
    "label_mlp_size": 128,
    "use_intratoken_heuristics": true,
    "embedding_dropout": 0.3,
    "encoded_dropout": 0.33,
    "upos_dropout": 0.33,
    "xpos_dropout": 0.33,
    "mlp_dropout": 0.33
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 32,
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