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

@DatasetReader.register('ud_char_level')
class UniversalDependenciesCharacterLevelDatasetReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            use_char_level_segmentation: bool = True,
            use_upos: bool = True,
            use_xpos: bool = False,
            use_deptags: bool = True,
            tokenizer: Tokenizer = None,
            root_token: str = ROOT_TOKEN,
            intratoken_tag: str = INTRATOKEN_TAG,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_xpos = use_xpos
        self.use_char_level_segmentation = use_char_level_segmentation
        self.use_upos = use_upos
        self.use_deptags = use_deptags
        self.tokenizer = tokenizer
        self.root_token = root_token
        self.intratoken_tag = intratoken_tag

    @staticmethod
    def read_corpus(filepath: str) -> Iterator[List[List[str]]]:
        with open(filepath, 'r') as f:
            for sentence in re.split(r'\n\n+', f.read()):
                if not sentence:
                    continue
                output = []
                for token in sentence.split('\n'):
                    if token.startswith('#') or not token:
                        continue
                    if re.match('\t', token):
                        output.append(token.split('\t'))
                    else:
                        output.append(token.split())
                yield output

    def convert_one_sentence(self, lines: List[List[str]]) -> List[List[str]]:

        if self.use_char_level_segmentation:

            token2char_id_mapping = {0: 0}

            moving_number = 1
            for line in lines:
                old_id = int(line[0])
                new_id = moving_number + len(line[1]) - 1
                token2char_id_mapping[old_id] = new_id
                moving_number += len(line[1])

            new_lines = [['0', self.root_token, self.root_token, 'ROOT', 'ROOT', '_', '0', 'ROOT', '_', '_']]
            moving_number = 1
            for line in lines:
                for i, symbol in enumerate(line[1]):
                    new_line = line[:]
                    new_line[0] = str(moving_number)
                    new_line[1] = symbol

                    if line[6] != '_':
                        old_head_id = int(line[6])

                        if i == len(line[1]) - 1:
                            new_line[6] = str(token2char_id_mapping[old_head_id])
                        else:
                            new_line[6] = str(moving_number + 1)
                            new_line[7] = self.intratoken_tag

                    new_lines.append(new_line)
                    moving_number += 1
        else:
            new_lines = [['0', self.root_token, self.root_token, 'ROOT', 'ROOT', '_', '0', 'ROOT', '_', '_']]

            for line in lines:
                new_lines.append(line)

        return new_lines

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        if self.use_char_level_segmentation:

            for sentence in self.read_corpus(file_path):
                converted_sentence = self.convert_one_sentence(sentence)

                chars = [x[1] for x in converted_sentence]

                if self.use_upos:
                    upos_tags = [x[3] for x in converted_sentence]
                else:
                    upos_tags = None

                if self.use_xpos:
                    xpos_tags = [x[4] for x in converted_sentence]
                else:
                    xpos_tags = None

                if self.use_deptags:
                    arc_indices = [(int(x[0]), int(x[6])) for x in converted_sentence if int(x[0]) > 0]
                    arc_tags = [x[7] for x in converted_sentence if x[7] != 'ROOT']
                else:
                    arc_indices = None
                    arc_tags = None

                yield self.text_to_instance(chars, arc_indices=arc_indices, arc_tags=arc_tags,
                                            upos_tags=upos_tags, xpos_tags=xpos_tags)
        else:

            corpus = self.read_corpus(file_path)
            output = convert_one_sentence(corpus)

            for x in output:

                chars = [x[1] for x in output if len(x[0]) < 3]

                if self.use_upos:
                    upos_tags = [x[3] for x in output if len(x[0]) < 3]
                else:
                    upos_tags = None

                if self.use_xpos:
                    xpos_tags = [x[4] for x in output if len(x[0]) < 3]
                else:
                    xpos_tags = None

                if self.use_deptags:
                    arc_indices = [(x[0], x[6]) for x in output if x[0] != 0 and len(x[0]) < 3]
                    arc_tags = [x[7] for x in output if len(x[0]) < 3 and x[7] != 'ROOT']
                else:
                    arc_indices = None
                    arc_tags = None

                yield self.text_to_instance(chars, arc_indices=arc_indices, arc_tags=arc_tags,
                                            upos_tags=upos_tags, xpos_tags=xpos_tags)

    @overrides
    def text_to_instance(
            self,
            chars: List[str],
            arc_indices: List[Tuple[int, int]] = None,
            arc_tags: List[str] = None,
            upos_tags: List[str] = None,
            xpos_tags: List[str] = None
    ) -> Instance:

        fields: Dict[str, Field] = {}

        tokens = self.tokenizer.tokenize(' '.join(chars)) \
            if self.tokenizer is not None else [Token(t) for t in chars]

        text_field = TextField(tokens, self._token_indexers)
        fields['chars'] = text_field

        if upos_tags is not None:
            fields['upos_tags'] = SequenceLabelField(upos_tags, text_field, label_namespace='upos')

        if xpos_tags is not None:
            fields['xpos_tags'] = SequenceLabelField(xpos_tags, text_field, label_namespace='xpos')

        if arc_indices is not None and arc_tags is not None:
            fields['adjacency_matrix'] = AdjacencyField(arc_indices, text_field, arc_tags,
                                                        label_namespace='dependency', padding_value=-1)

        return Instance(fields)
