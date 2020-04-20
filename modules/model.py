import numpy as np
from typing import Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LeakyReLU, Sequential, CrossEntropyLoss

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder

# TODO: rewrite these modules here
from fastNLP.models.biaffine_parser import ArcBiaffine, LabelBilinear


def _find_cycle(vertices, edges):
    r"""
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py
    """
    # TODO: understand what this does and rewrite the docstring
    _index = 0
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        nonlocal _index
        _indices[v] = _index
        _lowlinks[v] = _index
        _index += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


def _mst(scores):
    r"""
    with some modification to support parser output for MST decoding
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L692
    """
    # TODO: understand what this does and rewrite the docstring
    length = scores.shape[0]
    min_score = scores.min() - 1
    eye = np.eye(length)
    scores = scores * (1 - eye) + min_score * eye
    heads = np.argmax(scores, axis=1)
    heads[0] = 0
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / head_scores)]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(
            scores[roots, new_heads] / root_scores)]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set)
    vertices = {0, }
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = min_score
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / old_scores
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


@Model.register('char_biaffine')
class CharBiaffineParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 arc_mlp_size: int,
                 label_mlp_size: int,
                 dependency_namespace: str = 'dependency',
                 use_greedy_infer: bool = False,
                 upos_head: bool = True,
                 upos_namespace: str = 'upos',
                 xpos_head: bool = False,
                 xpos_namespace: str = 'xpos',
                 embedding_dropout: float = 0.5,
                 encoded_dropout: float = 0.5,
                 mlp_dropout: float = 0.5) -> None:
        # TODO: add clever layer initialization

        super().__init__(vocab)
        self.embedder = text_field_embedder
        self.encoder = encoder
        encoder_output_dim = encoder.get_output_dim()

        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.mlp = Sequential(Linear(encoder_output_dim, 2 * (arc_mlp_size + label_mlp_size)),
                              LeakyReLU(0.1), Dropout(mlp_dropout))
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.num_labels = vocab.get_vocab_size(dependency_namespace)
        self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size,
                                             self.num_labels, bias=True)
        self.use_greedy_infer = use_greedy_infer

        self.upos_head = Linear(encoder_output_dim, vocab.get_vocab_size(upos_namespace)) \
            if upos_head else None
        self.xpos_head = Linear(encoder_output_dim, vocab.get_vocab_size(xpos_namespace)) \
            if xpos_head else None
        self.tagging_loss = CrossEntropyLoss()

        self.embedding_dropout = Dropout(embedding_dropout)
        self.encoded_dropout = Dropout(encoded_dropout)

    @staticmethod
    def greedy_decoder(arc_matrix, mask=None):
        # TODO: understand what this does and rewrite the docstring
        r"""
        贪心解码方式, 输入图, 输出贪心解码的parsing结果, 不保证合法的构成树
        :param arc_matrix: [batch, seq_len, seq_len] 输入图矩阵
        :param mask: [batch, seq_len] 输入图的padding mask, 有内容的部分为 1, 否则为 0.
            若为 ``None`` 时, 默认为全1向量. Default: ``None``
        :return heads: [batch, seq_len] 每个元素在树中对应的head(parent)预测结果
        """
        _, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix + torch.diag(arc_matrix.new(seq_len).fill_(-np.inf))
        flip_mask = mask.eq(False)
        matrix.masked_fill_(flip_mask.unsqueeze(1), -np.inf)
        _, heads = torch.max(matrix, dim=2)
        if mask is not None:
            heads *= mask.long()
        return heads

    @staticmethod
    def mst_decoder(arc_matrix, mask=None):
        # TODO: understand what this does and rewrite the docstring
        r"""
        用最大生成树算法, 计算parsing结果, 保证输出合法的树结构
        :param arc_matrix: [batch, seq_len, seq_len] 输入图矩阵
        :param mask: [batch, seq_len] 输入图的padding mask, 有内容的部分为 1, 否则为 0.
            若为 ``None`` 时, 默认为全1向量. Default: ``None``
        :return heads: [batch, seq_len] 每个元素在树中对应的head(parent)预测结果
        """
        batch_size, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix.clone()
        ans = matrix.new_zeros(batch_size, seq_len).long()
        lens = (mask.long()).sum(1) if mask is not None else torch.zeros(batch_size) + seq_len
        for i, graph in enumerate(matrix):
            len_i = lens[i]
            ans[i, :len_i] = torch.as_tensor(_mst(graph.detach()[:len_i, :len_i].cpu().numpy()), device=ans.device)
        if mask is not None:
            ans *= mask.long()
        return ans

    @staticmethod
    def arc_loss(arc_pred: torch.Tensor,
                 label_pred: torch.Tensor,
                 arc_true: torch.LongTensor,
                 label_true: torch.LongTensor,
                 mask: torch.BoolTensor) -> torch.Tensor:
        """
        Compute loss for dependency parsing.
        :param arc_pred: [batch_size, seq_len, seq_len]
        :param label_pred: [batch_size, seq_len, n_tags]
        :param arc_true: [batch_size, seq_len]
        :param label_true: [batch_size, seq_len]
        :param mask: [batch_size, seq_len]
        :return: loss value
        """

        batch_size, seq_len = arc_true.shape
        flip_mask = (mask == 0)
        _arc_pred = arc_pred.masked_fill(flip_mask.unsqueeze(1), -float('inf'))

        arc_true_filled = arc_true.clone()
        arc_true_filled[:, 0].fill_(-1)
        label_true_filled = label_true.clone()
        label_true_filled[:, 0].fill_(-1)

        arc_nll = F.cross_entropy(_arc_pred.view(-1, seq_len),
                                  arc_true_filled.view(-1), ignore_index=-1)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)),
                                    label_true_filled.view(-1), ignore_index=-1)

        return arc_nll + label_nll

    def forward(self,
                chars: Dict[str, torch.LongTensor],
                arc_indices: torch.LongTensor,
                arc_tags: torch.LongTensor,
                upos_tags: torch.LongTensor = None,
                xpos_tags: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        # TODO: make this compatible with AdjacencyField instead of two SequenceLabelFields
        # TODO: add metric monitoring

        mask = get_text_field_mask(chars)
        output_dict = {'mask': mask}

        batch_size, seq_len = mask.shape
        embedded = self.embedding_dropout(self.embedder(chars))
        encoded = self.encoded_dropout(self.encoder(embedded, mask))

        accumulated_loss = 0
        if self.upos_head is not None:
            logits = self.upos_head(encoded)
            output_dict['upos_logits'] = logits
            if upos_tags is not None:
                loss = self.tagging_loss(logits.transpose(1, 2), upos_tags)
                output_dict['upos_loss'] = loss.item()
                accumulated_loss += loss

        if self.xpos_head is not None:
            logits = self.xpos_head(encoded)
            output_dict['xpos_logits'] = logits
            if xpos_tags is not None:
                loss = self.tagging_loss(logits.transpose(1, 2), xpos_tags)
                output_dict['xpos_loss'] = loss.item()
                accumulated_loss += loss

        arc_indices = arc_indices.long()

        mlp_output = self.mlp(encoded)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = mlp_output[:, :, :arc_sz], mlp_output[:, :, arc_sz:2 * arc_sz]
        label_dep, label_head = mlp_output[:, :, 2 * arc_sz:2 * arc_sz + label_sz], \
                                mlp_output[:, :, 2 * arc_sz + label_sz:]

        arc_preds = self.arc_predictor(arc_head, arc_dep)

        # Use gold or predicted arcs to predict labels.
        if arc_indices is None or not self.training:
            # Use greedy decoding in training.
            if self.training or self.use_greedy_infer:
                heads = self.greedy_decoder(arc_preds, mask)
            else:
                heads = self.mst_decoder(arc_preds, mask)
            head_preds = heads
        else:
            if arc_indices is None:
                heads = self.greedy_decoder(arc_preds, mask)
                head_preds = heads
            else:
                head_preds = None
                heads = arc_indices

        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=mask.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_preds = self.label_predictor(label_head, label_dep)  # [N, max_len, num_label]
        arange_index = torch.arange(1, seq_len + 1, dtype=torch.long, device=mask.device).unsqueeze(0) \
            .repeat(batch_size, 1)  # batch_size x max_len
        app_masks = heads.ne(arange_index)  # batch_size x max_len
        app_masks = app_masks.unsqueeze(2).repeat(1, 1, self.num_labels)
        app_masks[:, :, 1:] = 0
        label_preds = label_preds.masked_fill(app_masks, -np.inf)

        output_dict.update({
            'arc_preds': arc_preds,
            'label_preds': label_preds,
        })
        if head_preds is not None:
            output_dict['head_preds'] = head_preds

        if arc_indices is not None and arc_tags is not None:
            loss = self.arc_loss(arc_preds, label_preds, arc_indices, arc_tags, mask)
            output_dict['dependency_loss'] = loss.item()
            output_dict['loss'] = accumulated_loss + loss

        return output_dict
