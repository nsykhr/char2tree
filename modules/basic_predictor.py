import numpy as np
from itertools import chain
from overrides import overrides
from typing import List, Dict, Union, Any

from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict


from modules import UniversalDependenciesDatasetReader, JointTaggerParser


def vote(votes: List[Any]) -> Any:
    return max(set(votes), key=votes.count)


@Predictor.register('ud_basic_char_level')
class UniversalDependenciesBasicCharacterLevelPredictor(Predictor):
    def __init__(self,
                 model: JointTaggerParser,
                 dataset_reader: UniversalDependenciesDatasetReader,
                 use_intratoken_heuristics: bool = True):
        super().__init__(model=model, dataset_reader=dataset_reader)
        self.use_intratoken_heuristics = use_intratoken_heuristics

    @overrides
    def _json_to_instance(self, json_dict: dict) -> Instance:
        tokens = json_dict['tokens']
        return self._dataset_reader.text_to_instance(tokens)

    def _predict(self, json_dict: JsonDict) -> Dict[str, List[Union[str, int]]]:
        output = {}
        prediction = self.predict_json(json_dict)

        if 'upos_logits' in prediction:
            output['upos'] = [self._model.vocab.get_token_from_index(
                int(np.argmax(token_logits)),
                namespace=self._model.upos_namespace)
                for token_logits in prediction['upos_logits'][1:]]

        if 'xpos_logits' in prediction:
            output['xpos'] = [self._model.vocab.get_token_from_index(
                int(np.argmax(token_logits)),
                namespace=self._model.xpos_namespace)
                for token_logits in prediction['xpos_logits'][1:]]

        if 'head_preds' in prediction and 'label_preds' in prediction:
            output['heads'] = prediction['head_preds'][1:]

            output['labels'] = []
            for i, (head, token_logits) in enumerate(zip(output['heads'], prediction['label_preds'][1:])):
                sorted_labels = [self._model.vocab.get_token_from_index(
                    int(x), namespace=self._model.dependency_namespace)
                                    for x in np.argsort(token_logits)][::-1]

                best_label = sorted_labels[0]

                if self.use_intratoken_heuristics:
                    for label in sorted_labels:
                        if label in (self._dataset_reader.intratoken_tag, 'left_crcmfix') and head != i + 2:
                            continue
                        best_label = label
                        break

                output['labels'].append(best_label)

            # This crutch forbids the model the predict the left_crcmfix label if it doesn't predict
            # any incorporated tokens further in the sequence.
            output['labels'] = [label if any(l.startswtith('incorp:') for l in output['labels'][i + 1:])
                                else label.replace('left_crcmfix', 'app')
                                for i, label in enumerate(output['labels'])]

        return output

    @staticmethod
    def update_head_indices(predictions: Dict[str, List[str]]) -> None:
        char_id2token_id = {0: 0}
        moving_number = 1
        for i, token in enumerate(predictions['tokens']):
            for num in range(moving_number, moving_number + len(token)):
                char_id2token_id[num] = i + 1

            moving_number += len(token)

        predictions['heads'] = [str(char_id2token_id[head]) for head in predictions['heads']]

    def predict(self, json_dict: JsonDict) -> Dict[str, List[str]]:
        if not json_dict:
            return {}

        char_preds = self._predict(json_dict)
        output = {key: [] for key in chain(char_preds, ['tokens'])}

        # We need to merge characters connected with the 'app' dependency type into single tokens.
        if 'labels' in char_preds and \
                any(label == self._dataset_reader.intratoken_tag
                    for label in char_preds['labels']):
            chars = json_dict['tokens'][1:]

            curr_token = ''
            curr_upos, curr_xpos = [], []

            for i, label in enumerate(char_preds['labels']):
                curr_token += chars[i]
                if 'upos' in char_preds:
                    curr_upos.append(char_preds['upos'][i])
                if 'xpos' in char_preds:
                    curr_xpos.append(char_preds['xpos'][i])

                if label == self._dataset_reader.intratoken_tag:
                    continue

                output['tokens'].append(curr_token)
                curr_token = ''

                if curr_upos:
                    output['upos'].append(vote(curr_upos))
                    curr_upos = []

                if curr_xpos:
                    output['xpos'].append(vote(curr_xpos))
                    curr_xpos = []

                output['heads'].append(char_preds['heads'][i])
                output['labels'].append(label)

            self.update_head_indices(output)

        else:
            char_preds.update({'tokens': json_dict['tokens'][1:]})
            char_preds['heads'] = list(map(str, char_preds['heads']))
            output = char_preds

        return output
