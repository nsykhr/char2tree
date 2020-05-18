{
  "dataset_reader": {
    "type": "ud_reader"
  },
  "train_data_path": "data/Chukchi/ckt-train.conllu",
  "vocabulary": {
    "non_padded_namespaces": ["*tags", "*labels", "upos", "dependency"]
  },
  "model": {
    "type": "char_level_joint",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 50,
      "hidden_size": 128,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.25,
      "layer_dropout_probability": 0.25,
      "use_highway": true
    },
    "upos_hidden": 64,
    "arc_mlp_size": 128,
    "label_mlp_size": 64,
    "use_intratoken_heuristics": true,
    "embedding_dropout": 0.2,
    "encoded_dropout": 0.25,
    "upos_dropout": 0.25,
    "mlp_dropout": 0.25
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 16,
    "biggest_batch_first": true,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 5e-3,
      "weight_decay": 1e-4
    },
    "num_epochs": 50,
    "grad_norm": 5.0,
    "cuda_device": 0
  }
}