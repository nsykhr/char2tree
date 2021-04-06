import re
from overrides import overrides
from typing import List, Tuple, Dict, Iterator

from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, SequenceLabelField, AdjacencyField

ROOT_TOKEN = '<ROOT>'
INTRATOKEN_TAG = 'app'


@DatasetReader.register('ud_reader')
class UniversalDependenciesDatasetReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            character_level: bool = True,
            use_upos: bool = True,
            use_xpos: bool = False,
            use_deptags: bool = True,
            num_cols: int = 10,
            tokenizer: Tokenizer = None,
            use_lowercase: bool = True,
            root_token: str = ROOT_TOKEN,
            intratoken_tag: str = INTRATOKEN_TAG,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.character_level = character_level
        self.use_xpos = use_xpos
        self.use_upos = use_upos
        self.use_deptags = use_deptags

        if num_cols < 4:
            raise ValueError('There must be at least 4 columns in the CONLLU file.')
        self.num_cols = num_cols

        self.tokenizer = tokenizer
        self.use_lowercase = use_lowercase
        self.root_token = root_token
        self.intratoken_tag = intratoken_tag

    def read_corpus(self, filepath: str) -> Iterator[List[List[str]]]:
        with open(filepath, 'r') as f:
            for sentence in re.split(r'\n\n+', f.read()):
                if not sentence:
                    continue
                output = []
                for line in sentence.split('\n'):
                    if line.startswith('#') or not line:
                        continue
                    splitted_line = line.split('\t')
                    assert len(splitted_line) == self.num_cols, \
                        f'The number of columns must be {self.num_cols}, {len(splitted_line)} found.'
                    output.append(splitted_line)
                yield output

    @staticmethod
    def get_mapping(sentence: List[List[str]]) -> Dict[str, str]:
        token2char_id_mapping = {'0': '0'}

        moving_number = 1
        multiword_indices = set()

        for i, line in enumerate(sentence):
            if '-' in line[0]:  # multi-word token
                multiword_indices = set(line[0].split('-'))
                continue

            old_id = line[0]
            new_id = str(moving_number + len(line[1]) - 1)
            token2char_id_mapping[old_id] = new_id

            moving_number += len(line[1])

            if line[6] in multiword_indices:
                line[7] = 'fused:' + line[7] if not line[7].startswith('fused:') else line[7]
                multiword_indices = set()

        return token2char_id_mapping

    def convert_to_character_level(self, sentence: List[List[str]]) -> List[List[str]]:
        token2char_id_mapping = self.get_mapping(sentence)

        new_sentence = [['0', self.root_token, self.root_token, 'ROOT', 'ROOT', '_', '0', 'ROOT', '_', '_']]
        moving_number = 1
        for i, line in enumerate(sentence):
            token_idx = line[0]

            if not token_idx.isdigit():
                continue

            for j, symbol in enumerate(line[1]):
                new_line = line[:]
                new_line[0] = str(moving_number)
                new_line[1] = symbol

                if j == len(line[1]) - 1:
                    if line[6] != '_':
                        old_head_id = line[6]
                        new_line[6] = token2char_id_mapping[old_head_id]
                else:
                    new_line[6] = str(moving_number + 1)
                    new_line[7] = self.intratoken_tag

                new_sentence.append(new_line)
                moving_number += 1

        return new_sentence

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        for sentence in self.read_corpus(file_path):
            if any(not x[0].isdigit() or not x[6].isdigit() for x in sentence):
                continue

            if self.character_level:
                converted_sentence = self.convert_to_character_level(sentence)
            else:
                converted_sentence = [
                    ['0', self.root_token, self.root_token, 'ROOT', 'ROOT', '_', '0', 'ROOT', '_', '_']
                ] + sentence

            tokens = [x[1] for x in converted_sentence]

            upos_tags = [x[3] for x in converted_sentence] if self.use_upos else None
            xpos_tags = [x[4] for x in converted_sentence] if self.use_xpos else None

            arc_indices = [(int(x[0]), int(x[6])) for x in converted_sentence if int(x[0]) > 0] \
                if self.use_deptags else None
            arc_tags = [x[7] for x in converted_sentence if x[7] != 'ROOT'] if self.use_deptags else None

            yield self.text_to_instance(tokens, arc_indices=arc_indices, arc_tags=arc_tags,
                                        upos_tags=upos_tags, xpos_tags=xpos_tags)

    @overrides
    def text_to_instance(
            self,
            tokens: List[str],
            arc_indices: List[Tuple[int, int]] = None,
            arc_tags: List[str] = None,
            upos_tags: List[str] = None,
            xpos_tags: List[str] = None
    ) -> Instance:

        fields: Dict[str, Field] = {}

        if self.use_lowercase:
            tokens = list(map(str.lower, tokens))
        tokens = self.tokenizer.tokenize(' '.join(tokens)) \
            if self.tokenizer is not None else [Token(t) for t in tokens]

        text_field = TextField(tokens, self._token_indexers)
        fields['tokens'] = text_field

        if upos_tags is not None:
            fields['upos_tags'] = SequenceLabelField(upos_tags, text_field, label_namespace='upos')

        if xpos_tags is not None:
            fields['xpos_tags'] = SequenceLabelField(xpos_tags, text_field, label_namespace='xpos')

        if arc_indices is not None and arc_tags is not None:
            fields['adjacency_matrix'] = AdjacencyField(arc_indices, text_field, arc_tags,
                                                        label_namespace='dependency', padding_value=-1)

        return Instance(fields)
