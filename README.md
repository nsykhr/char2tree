# End-to-End Segmentator, Part-of-Speech Tagger, and Dependency Parser for Universal Dependencies

In this repository you will find the code for our paper on joint segmenting, tagging, and parsing of polysynthetic languages. While developed with these languages in mind (and thus lacking some heuristics that may slightly improve performance on standard European languages), the code works for any data in the Universal Dependencies format.

To run training, create or choose a configuration file that encodes the model's hyperparameters such as the tasks you want the model to solve (any of the three tasks can be switched on and off), the number of parameters in the layers, and regularization. Then open the repository's directory in the Terminal and type:

`allennlp train <CONFIG_PATH> -s <SERIALIZATION_DIR> --include-package modules`.

Our approach is inspired by *A Graph-based Model for Joint Chinese Word Segmentation and Dependency Parsing*, a 2019 paper by Hang Yan, Xipeng Qiu, and Xuanjing Huang. Some of the core parts of the model's code were adapted from their open-sourced code for the paper, available at https://github.com/fastnlp/JointCwsParser.