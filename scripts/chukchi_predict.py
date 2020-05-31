from pathlib import Path

from allennlp.models import Model
from allennlp.data import DatasetReader
from allennlp.common.params import Params
from modules import UniversalDependenciesCharacterLevelPredictor

from predict import get_predictions, save_predictions_to_conllu


def main():
    for i in range(10):
        model_path = f'../models/chukchi/chukchi_{i}'
        test_path = f'../data/Chukchi/flat_test/ckt_hse-ud-test.0{i}.conllu'
        savepath = f'../results/chukchi/test_result_{i}.conllu'

        config = Params.from_file(Path(model_path, 'config.json'))
        dataset_reader = DatasetReader.from_params(config['dataset_reader'])

        model = Model.load(config, serialization_dir=model_path, cuda_device=-1)
        model.eval()

        predictor = UniversalDependenciesCharacterLevelPredictor(model=model,
                                                                 dataset_reader=dataset_reader)

        predictions = get_predictions(test_path, dataset_reader, predictor)
        save_predictions_to_conllu(savepath, predictions)


if __name__ == '__main__':
    main()
