# End-to-End Segmentator, Part-of-Speech Tagger, and Dependency Parser for Universal Dependencies

In this repository you will find the code for our paper on joint segmenting, tagging, and parsing of incorporating languages (namely, we've been using Chukchi, for which we manually annotated a small treebank). While developed with these languages in mind, the code works for any data in the Universal Dependencies format.

We've decided to use the AllenNLP library for model creation because of its versatility and configurability, as well as because it uses pyTorch as its backend deep learning framework. To run training, create or choose a configuration file that encodes the model's hyperparameters such as the tasks you want the model to solve (any of the three tasks can be switched on and off), the number of parameters in the layers, and regularization. Then open the repository's directory in the Terminal and type:

`allennlp train <CONFIG_PATH> -s <SERIALIZATION_DIR> --include-package modules`.

Furthermore, you can use the `training_example.ipynb` notebook as reference.

Our approach is inspired by *A Graph-based Model for Joint Chinese Word Segmentation and Dependency Parsing*, a 2019 paper by Hang Yan, Xipeng Qiu, and Xuanjing Huang. Some of the core parts of the model's code were adapted from their open-sourced code for the paper, available at https://github.com/fastnlp/JointCwsParser. The resulting architecture is similar to the one implemented in the paper *An improved neural network model for joint POS tagging and dependency parsing* by Dat Quoc Nguyen and Karin Verspoor (the code is available at https://github.com/datquocnguyen/jPTDP, it's written in a deep learning framework called _dynet_).

The important feature that distinguishes our work from those related works is the format of data annotation for incorporating languages that we've developed for the Chukchi treebank.

Team members: Francis Tyers, Nikita Sykhrannov, Elizaveta Ezhergina, Karina Mischenkova
