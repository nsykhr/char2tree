"""
This script is intended to flatten the Chukchi testing data for evaluation with the CONLL evaluation script.
"""
import re
import argparse
from tqdm import tqdm
from pathlib import Path

from typing import List, Tuple, Dict, Iterator, Union


def read_corpus(filepath: Union[str, Path]) -> Iterator[List[List[str]]]:
    with open(filepath, 'r') as f:
        for sentence in re.split(r'\n\n+', f.read()):
            if not sentence:
                continue
            output = []
            for line in sentence.split('\n'):
                if not line:
                    continue
                if line.startswith('#'):
                    output.append([line])
                    continue
                output.append(line.split('\t'))
            yield output


def get_token_mapping(sentence: List[List[str]]) -> Dict[str, str]:
    token_mapping: Dict[str, str] = {'0': '0'}
    shift, shift_buffer, accumulated_length = 0, 0, 0

    for i, line in enumerate(sentence):
        if '-' in line[0] or ('.' in line[0] and not line[1] in sentence[i - int(line[0].split('.')[1])][1]):
            shift -= 1
            continue

        if '.' in line[0]:
            subtoken_start = sentence[i - int(line[0].split('.')[1])][1].find(line[1])

            if subtoken_start - accumulated_length == 0:
                token_mapping[line[0].split('.')[0]] = str(int(token_mapping[line[0].split('.')[0]]) + 1)
            else:
                token_mapping[line[0].split('.')[0]] = str(int(token_mapping[line[0].split('.')[0]]) + 2)
                token_mapping[line[0]] = str(i + shift + 1)
                shift_buffer += 1
                accumulated_length += subtoken_start + len(line[1])
                continue

            accumulated_length += subtoken_start + len(line[1])

            if subtoken_start == 0:
                token_mapping[line[0]] = str(i + shift)
                continue

        else:
            accumulated_length = 0
            shift += shift_buffer
            shift_buffer = 0

        token_mapping[line[0]] = str(i + shift + 1)

    return token_mapping


def flatten_sentence(sentence: List[List[str]]) -> Tuple[List[List[str]], List[str]]:
    metadata = [line[0] for line in sentence if len(line) == 1]
    sentence = sentence[len(metadata):]

    shift = 0
    new_sentence = []
    multiword_indices = set()
    token_mapping = get_token_mapping(sentence)

    for i, line in enumerate(sentence):
        if '.' in line[0] and not line[1] in sentence[i - int(line[0].split('.')[1])][1]:
            shift -= 1
            continue

        if '-' in line[0]:
            shift -= 1
            multiword_indices = set(line[0].split('-'))
            continue

        new_line = line[:]
        new_line[0] = str(i + shift + 1)
        new_line[8] = '_'

        if '.' in line[0]:
            old_head_id, rel_type = line[8].split(':')
            new_line[6] = token_mapping[old_head_id]
            new_line[7] = 'incorp:' + rel_type

            subtoken_start = new_sentence[-1][1].find(line[1])
            subtoken_end = subtoken_start + len(line[1])

            left_token_part = new_sentence[-1][:]

            if subtoken_start > 0 and not \
                    new_sentence[-1][7].startswith('incorp:'):
                new_sentence[-1][1] = \
                    new_sentence[-1][1][:subtoken_start]
                new_sentence[-1][6] = \
                    str(int(new_sentence[-1][0]) + 1)
                new_sentence[-1][7] = 'left_crcmfix'

            elif subtoken_start == 0:
                new_line[0] = str(int(new_line[0]) - 1)
                del new_sentence[-1]
                shift -= 1

            new_sentence.append(new_line)

            left_token_part[0] = str(i + shift + 2)
            left_token_part[1] = left_token_part[1][subtoken_end:]
            new_sentence.append(left_token_part)
            shift += 1

        else:
            new_line[6] = token_mapping[line[6]]

            if line[6] in multiword_indices:
                new_line[7] = 'fused:' + new_line[7]
                multiword_indices = set()

            new_sentence.append(new_line)

    return new_sentence, metadata


def flatten_file(src_path: Union[str, Path], tgt_path: Union[str, Path]) -> None:
    with open(tgt_path, 'w') as f:
        for sentence in read_corpus(src_path):
            flat_sentence, metadata = flatten_sentence(sentence)

            text = ''.join([x[1] for x in flat_sentence])
            for i, element in enumerate(metadata):
                if element.startswith('# text ='):
                    metadata[i] = f'# text = {text}'

            f.write('\n'.join(metadata) + '\n')
            f.write('\n'.join(['\t'.join(line) for line in flat_sentence]) + '\n\n')


def flatten_dir(src_dir: Union[str, Path], tgt_dir: Union[str, Path], keyword: str) -> None:
    assert src_dir != tgt_dir, 'Source and target paths are the same.'

    for entry in tqdm(list(Path(src_dir).iterdir())):
        if entry.name.endswith('.conllu') and keyword in entry.name:
            flatten_file(entry, Path(tgt_dir, entry.name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', '-s', type=str)
    parser.add_argument('--tgt_dir', '-t', type=str)
    parser.add_argument('--keyword', '-k', type=str, default='test')
    args = parser.parse_args()

    flatten_dir(args.src_dir, args.tgt_dir, args.keyword)


if __name__ == '__main__':
    main()
