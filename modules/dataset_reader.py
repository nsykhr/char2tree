import re
from overrides import overrides
from typing import List, Tuple, Dict, Iterator

from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, SequenceLabelField, AdjacencyField

INTRATOKEN_TAG = 'app'


@DatasetReader.register('ud_char_level')
class UniversalDependenciesCharacterLevelDatasetReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            use_upos: bool = False,
            use_xpos: bool = False,
            use_deptags: bool = False,
            tokenizer: Tokenizer = None,
            intratoken_tag: str = INTRATOKEN_TAG,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_xpos = use_xpos
        self.use_upos = use_upos
        self.use_deptags = use_deptags
        self.tokenizer = tokenizer
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
                    output.append(token.split('\t'))
                yield output

    def convert_one_sentence(self, lines: List[List[str]]) -> List[List[str]]:
        token2char_id_mapping = {0: 0}

        moving_number = 1
        for line in lines:
            old_id = int(line[0])
            new_id = moving_number + len(line[1]) - 1
            token2char_id_mapping[old_id] = new_id
            moving_number += len(line[1])

        new_lines = []
        moving_number = 1
        for line in lines:
            for i, symbol in enumerate(line[1]):
                new_line = line[:]
                new_line[0] = str(moving_number)
                moving_number += 1
                new_line[1] = symbol

                if line[6] != '_':
                    old_head_id = int(line[6])

                    if i == len(line[1]) - 1:
                        new_line[6] = str(token2char_id_mapping[old_head_id])
                    else:
                        new_line[6] = str(moving_number + 1)
                        new_line[7] = self.intratoken_tag

                new_lines.append(new_line)

        return new_lines

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
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
                # The commented version is for SequenceLabelField which we are not using anymore.
                # arc_indices = [int(x[6]) for x in converted_sentence]
                arc_indices = [(int(x[0]), int(x[6])) for x in converted_sentence]
                arc_tags = [x[7] for x in converted_sentence]
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
            # arc_indices: List[int] = None,
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
            # fields['arc_tags'] = SequenceLabelField(arc_tags, text_field, label_namespace='dependency')
            # fields['arc_indices'] = SequenceLabelField(arc_indices, text_field, label_namespace='arc_indices')
            fields['adjacency_matrix'] = AdjacencyField(arc_indices, text_field, arc_tags,
                                                        label_namespace='dependency', padding_value=0)

        return Instance(fields)
