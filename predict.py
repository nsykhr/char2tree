import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from allennlp.models import Model
from allennlp.data import DatasetReader
from allennlp.common.params import Params

from modules import UniversalDependenciesCharacterLevelDatasetReader, \
    UniversalDependenciesBasicCharacterLevelPredictor


def save_predictions_to_conllu(savepath: str, predictions: List[Dict[str, List[str]]]) -> None:
    with open(savepath, 'w') as f:
        for i, sentence in enumerate(predictions):
            f.write(f'# sent_id = test-s{i+1}\n')

            text = ''.join(sentence['tokens'])
            f.write(f'# text = {text}\n')

            for j, token in enumerate(sentence['tokens']):
                upos = sentence['upos'][j] if 'upos' in sentence else '_'
                xpos = sentence['xpos'][j] if 'xpos' in sentence else '_'
                head = sentence['heads'][j] if 'heads' in sentence else '_'
                label = sentence['labels'][j] if 'labels' in sentence else '_'

                f.write('\t'.join([str(j+1), token, '_', upos, xpos, '_', head, label, '_', '_']) + '\n')

            f.write('\n')


def get_predictions(test_path: str, dataset_reader: UniversalDependenciesCharacterLevelDatasetReader,
                    predictor: UniversalDependenciesBasicCharacterLevelPredictor) -> List[Dict[str, List[str]]]:
    all_predictions = []
    for sentence in tqdm(dataset_reader.read_corpus(test_path)):
        tokens = [dataset_reader.root_token] + list(''.join([x[1] for x in sentence]))
        json_dict = {'tokens': tokens}
        predictions = predictor.predict(json_dict)
        all_predictions.append(predictions)

    return all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', '-t', type=str, help='Path to the testing data file in the UD format.')
    parser.add_argument('--model-path', '-m', type=str, help='Path to the directory with the trained model.')
    parser.add_argument('--savepath', '-s', type=str, help='Path to save the CONLLU file with predictions.')
    parser.add_argument('--cuda', '-c', type=int, default=-1, help='CUDA device index (default value is -1 for CPU).')
    args = parser.parse_args()

    config = Params.from_file(Path(args.model_path, 'config.json'))
    dataset_reader = DatasetReader.from_params(config['dataset_reader'])

    model = Model.load(config, serialization_dir=args.model_path, cuda_device=args.cuda)
    model.eval()

    predictor = UniversalDependenciesBasicCharacterLevelPredictor(model=model, dataset_reader=dataset_reader)

    predictions = get_predictions(args.test_path, dataset_reader, predictor)
    save_predictions_to_conllu(args.savepath, predictions)


if __name__ == '__main__':
    main()
