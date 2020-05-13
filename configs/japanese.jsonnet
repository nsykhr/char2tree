{
  "dataset_reader": {
    "type": "ud_char_level",
    "use_xpos": true
  },
  "train_data_path": "data/japanese/ja_gsd-ud-train.conllu",
  "validation_data_path": "data/japanese/ja_gsd-ud-dev.conllu",
  "vocabulary": {
    "non_padded_namespaces": ["*tags", "*labels", "upos", "xpos", "dependency"]
  },
  "model": {
    "type": "char_biaffine",
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
      "hidden_size": 512,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.25,
      "layer_dropout_probability": 0.25,
      "use_highway": true
    },
    "arc_mlp_size": 512,
    "label_mlp_size": 128,
    "use_greedy_infer": false,
    "xpos_head": true,
    "embedding_dropout": 0.25,
    "encoded_dropout": 0.25,
    "mlp_dropout": 0.25
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 352,
    "biggest_batch_first": true,
    "sorting_keys": [["chars", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 5e-4,
      "weight_decay": 1e-4
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "patience": 2,
      "min_lr": 1e-5
    },
    "num_epochs": 100,
    "patience": 10,
    "grad_norm": 5.0,
    "cuda_device": 0
  }
}