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
    def get_token2char_id_mapping(sentence: List[List[str]]) -> Tuple[Dict[int, int],
                                                                      Dict[int, Dict[int, List[str]]]]:
        """
        This function handles both creating the mapping and processing the "special cases":
        multi-word tokens and enhanced dependencies (both signify an incorporation of some kind).
        It does the latter by modifying the sentence in-place.
        """
        enhanced_dependencies = {}
        token2char_id_mapping = {0: 0}

        moving_number = 1
        multiword_indices = set()

        for i, line in enumerate(sentence):
            if '-' in line[0]:  # multi-word token
                multiword_indices = set(line[0].split('-'))
                continue

            if '.' in line[0]:  # enhanced dependency
                base_token_idx = i - int(line[0].split('.')[-1])
                subtoken_start = sentence[base_token_idx][1].find(line[1])
                if subtoken_start == -1:
                    print(f'Enhanced dependencies error: subtoken {line[1]} not found',
                          f'in the base token {sentence[base_token_idx][1]}.')
                    continue

                base_token_idx += 1  # To account for 1-indexing in the CONLLU.
                enhanced_dependencies[base_token_idx] = {}

                subtoken_end = subtoken_start + len(line[1])

                for char_idx in range(subtoken_start, subtoken_end):
                    enhanced_dependencies[base_token_idx][char_idx] = line[:]
                    if char_idx == subtoken_start:
                        enhanced_dependencies[base_token_idx][char_idx].append('<SUBTOKEN_START>')
                    if char_idx == subtoken_end - 1:
                        enhanced_dependencies[base_token_idx][char_idx].append('<SUBTOKEN_END>')

                continue

            old_id = int(line[0])
            new_id = moving_number + len(line[1]) - 1
            token2char_id_mapping[old_id] = new_id
            moving_number += len(line[1])

            if line[6] in multiword_indices:
                line[7] = 'fused:' + line[7]
                multiword_indices = set()

        return token2char_id_mapping, enhanced_dependencies

    def convert_to_character_level(self, sentence: List[List[str]]) -> List[List[str]]:
        token2char_id_mapping, enhanced_dependencies = self.get_token2char_id_mapping(sentence)

        new_sentence = [['0', self.root_token, self.root_token, 'ROOT', 'ROOT', '_', '0', 'ROOT', '_', '_']]
        moving_number = 1
        for line in sentence:
            if not line[0].isdigit():
                continue

            token_idx = int(line[0])

            for i, symbol in enumerate(line[1]):
                new_line = line[:]
                new_line[0] = str(moving_number)
                new_line[1] = symbol

                if i == len(line[1]) - 1:
                    if line[6] != '_':
                        old_head_id = int(line[6])
                        new_line[6] = str(token2char_id_mapping[old_head_id])
                else:
                    new_line[6] = str(moving_number + 1)
                    new_line[7] = self.intratoken_tag

                if token_idx in enhanced_dependencies and i in enhanced_dependencies[token_idx]:
                    if i > 0 and enhanced_dependencies[token_idx][i][-1] == '<SUBTOKEN_START>':
                        # The first character of an incorporated subtoken is the head
                        # of the previous character with the special relation type left_crcmfix.
                        new_sentence[-1][6] = str(moving_number)
                        new_sentence[-1][7] = 'left_crcmfix'

                    if enhanced_dependencies[token_idx][i][-1] == '<SUBTOKEN_END>':
                        # The last character of an incorporated subtoken is the child
                        # of the last character of the whole token.
                        old_head_id, rel_type = enhanced_dependencies[token_idx][i][8].split(':')
                        new_line[6] = str(token2char_id_mapping[int(old_head_id)])
                        new_line[7] = 'incorp:' + rel_type

                    new_line[3] = enhanced_dependencies[token_idx][i][3]
                    new_line[4] = enhanced_dependencies[token_idx][i][4]

                new_sentence.append(new_line)
                moving_number += 1

        return new_sentence

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        for sentence in self.read_corpus(file_path):
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
