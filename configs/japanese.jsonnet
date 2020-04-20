local EMBEDDING_DIM = 300;

{
  "dataset_reader": {
    "type": "ud_char_level",
    "use_xpos": true
  },
  "train_data_path": "data/Japanese/ja_gsd-ud-train.conllu",
  "validation_data_path": "data/Japanese/ja_gsd-ud-dev.conllu",
  "vocabulary": {
    "non_padded_namespaces": ["*tags", "*labels", "upos", "xpos", "dependency"]
  },
  "model": {
    "type": "char_biaffine",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": EMBEDDING_DIM,
      "hidden_size": 512,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.25,
      "layer_dropout_probability": 0.25,
      "use_highway": true
    },
    "arc_mlp_size": 512,
    "label_mlp_size": 128,
    "use_greedy_infer": true,
    "xpos_head": true,
    "embedding_dropout": 0.25,
    "encoded_dropout": 0.25,
    "mlp_dropout": 0.25
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
      "lr": 0.01,
      "weight_decay": 1e-5
    },
    "learning_rate_scheduler": {
      "type": "step",
      "gamma": 0.5,
      "step_size": 5
    },
    "num_epochs": 100,
    "patience": 10,
    "grad_norm": 5.0,
    "cuda_device": 0
  }
}