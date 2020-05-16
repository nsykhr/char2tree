from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, PReLU, Sequential

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from modules.utils import mst
from modules.biaffine_parser import ArcBiaffine, LabelBilinear


@Model.register('char_level_joint')
class CharacterLevelJointModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 upos_head: bool = True,
                 upos_namespace: str = 'upos',
                 upos_hidden: int = 256,
                 xpos_head: bool = False,
                 xpos_namespace: str = 'xpos',
                 xpos_hidden: int = 256,
                 dependency_head: bool = True,
                 dependency_namespace: str = 'dependency',
                 arc_mlp_size: int = 512,
                 label_mlp_size: int = 128,
                 use_greedy_infer: bool = False,
                 use_intratoken_heuristics: bool = False,
                 embedding_dropout: float = 0.5,
                 encoded_dropout: float = 0.5,
                 upos_dropout: float = 0.5,
                 xpos_dropout: float = 0.5,
                 mlp_dropout: float = 0.5) -> None:
        super().__init__(vocab)
        self.embedder = text_field_embedder
        self.encoder = encoder
        encoder_output_dim = encoder.get_output_dim()

        self.upos_head = Sequential(Linear(encoder_output_dim, upos_hidden),
                                    PReLU(), Dropout(upos_dropout),
                                    Linear(upos_hidden, vocab.get_vocab_size(upos_namespace))) \
            if upos_head else None

        self.xpos_head = Sequential(Linear(encoder_output_dim, xpos_hidden),
                                    PReLU(), Dropout(xpos_dropout),
                                    Linear(xpos_hidden, vocab.get_vocab_size(xpos_namespace))) \
            if xpos_head else None

        if dependency_head:
            self.arc_mlp_size = arc_mlp_size
            self.label_mlp_size = label_mlp_size
            self.mlp = Sequential(Linear(encoder_output_dim, 2 * (arc_mlp_size + label_mlp_size)),
                                  PReLU(), Dropout(mlp_dropout))
            self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
            self.num_labels = vocab.get_vocab_size(dependency_namespace)
            self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size,
                                                 self.num_labels, bias=True)
            self.use_greedy_infer = use_greedy_infer
            self.use_intratoken_heuristics = use_intratoken_heuristics
        else:
            self.arc_mlp_size = None
            self.label_mlp_size = None
            self.mlp = None
            self.arc_predictor = None
            self.num_labels = None
            self.label_predictor = None
            self.use_greedy_infer = None
            self.use_intratoken_heuristics = None

        self.embedding_dropout = Dropout(embedding_dropout)
        self.encoded_dropout = Dropout(encoded_dropout)

        self.upos_namespace = upos_namespace
        self.xpos_namespace = xpos_namespace
        self.dependency_namespace = dependency_namespace

    @staticmethod
    def greedy_decoder(arc_matrix: torch.Tensor, mask: torch.Tensor = None):
        r"""
        Greedy decoding method. We simply choose the head index with the highest probability, therefore,
        legal tree structure is not guaranteed.

        :param arc_matrix: [batch, seq_len, seq_len]
        :param mask: [batch, seq_len]
        :return heads: [batch, seq_len]: prediction result - head of each token in the sequence
        """
        batch_size, seq_len = arc_matrix.size()[:2]
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=arc_matrix.device)
        flipped_mask = (mask == 0)

        matrix = arc_matrix + torch.diag(arc_matrix.new_zeros(seq_len).fill_(-float('inf')))
        matrix.masked_fill_(flipped_mask.unsqueeze(1), -float('inf'))

        heads = torch.max(matrix, dim=2)[1]
        heads *= mask.long()

        return heads

    @staticmethod
    def mst_decoder(arc_matrix: torch.Tensor, mask: torch.Tensor = None):
        r"""
        Use the maximum spanning tree algorithm to calculate the parsing result
        and ensure that the output is a legal tree structure.

        :param arc_matrix: [batch, seq_len, seq_len]
        :param mask: [batch, seq_len]
        :return heads: [batch, seq_len]: prediction result - head of each token in the sequence
        """
        batch_size, seq_len = arc_matrix.size()[:2]
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=arc_matrix.device)

        matrix = arc_matrix.clone()
        ans = matrix.new_zeros(batch_size, seq_len).long()
        lengths = (mask.long()).sum(1) if mask is not None \
            else torch.zeros(batch_size, dtype=torch.long, device=arc_matrix.device) + seq_len

        for i, graph in enumerate(matrix):
            len_i = lengths[i]
            ans[i, :len_i] = torch.as_tensor(mst(graph.detach()[:len_i, :len_i].cpu().numpy()), device=ans.device)

        ans *= mask.long()

        return ans

    @staticmethod
    def arc_loss(arc_pred: torch.Tensor,
                 label_pred: torch.Tensor,
                 arc_true: torch.Tensor,
                 label_true: torch.Tensor,
                 flipped_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for dependency parsing.
        :param arc_pred: [batch_size, seq_len, seq_len]
        :param label_pred: [batch_size, seq_len, num_labels]
        :param arc_true: [batch_size, seq_len]
        :param label_true: [batch_size, seq_len]
        :param flipped_mask: [batch_size, seq_len]
        :return: loss value
        """
        batch_size, seq_len = arc_true.shape
        masked_arc_pred = arc_pred.masked_fill(flipped_mask.unsqueeze(1), -float('inf'))

        # The first token is always the <ROOT> token. We do not take the predictions for this token into account.
        arc_true_filled = arc_true.clone()
        arc_true_filled[:, 0].fill_(-1)
        label_true_filled = label_true.clone()
        label_true_filled[:, 0].fill_(-1)

        arc_nll = F.cross_entropy(masked_arc_pred.view(-1, seq_len),
                                  arc_true_filled.view(-1), ignore_index=-1)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)),
                                    label_true_filled.view(-1), ignore_index=-1)

        return arc_nll + label_nll

    @staticmethod
    def _transform_adjacency_matrix(adjacency_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param adjacency_matrix: a tensor of size [batch_size, seq_len, seq_len],
        where elements that are >= 0 signify an index of a syntactic tag
        :return: a tuple of tensors, arc_indices [batch_size, seq_len] and arc_tags [batch_size, seq_len]

        We use -1 padding for both arc indices and arc tags.
        """
        batch_size, seq_len = adjacency_matrix.size()[:2]
        arc_indices = -torch.ones((batch_size, seq_len), dtype=torch.long, device=adjacency_matrix.device)
        arc_tags = -torch.ones((batch_size, seq_len), dtype=torch.long, device=adjacency_matrix.device)

        nonzero_indices = torch.nonzero(adjacency_matrix + 1)
        for (b, i, j) in nonzero_indices:
            arc_indices[b, i] = j
            arc_tags[b, i] = adjacency_matrix[b, i, j]

        return arc_indices, arc_tags

    def forward(self,
                chars: Dict[str, torch.Tensor],
                upos_tags: torch.Tensor = None,
                xpos_tags: torch.Tensor = None,
                adjacency_matrix: torch.Tensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        # TODO: add online metric monitoring

        mask = get_text_field_mask(chars)
        output_dict = {'mask': mask}
        flipped_mask = (mask == 0)

        batch_size, seq_len = mask.shape
        embedded = self.embedding_dropout(self.embedder(chars))
        encoded = self.encoded_dropout(self.encoder(embedded, mask))

        accumulated_loss = torch.tensor(0., device=mask.device)
        if self.upos_head is not None:
            logits = self.upos_head(encoded)
            output_dict['upos_logits'] = logits
            if upos_tags is not None:
                masked_upos_tags = upos_tags.masked_fill(flipped_mask, -1)
                loss = F.cross_entropy(logits.transpose(1, 2), masked_upos_tags,
                                       ignore_index=-1)
                output_dict['upos_loss'] = loss.item()
                accumulated_loss += loss

        if self.xpos_head is not None:
            logits = self.xpos_head(encoded)
            output_dict['xpos_logits'] = logits
            if xpos_tags is not None:
                masked_xpos_tags = xpos_tags.masked_fill(flipped_mask, -1)
                loss = F.cross_entropy(logits.transpose(1, 2), masked_xpos_tags,
                                       ignore_index=-1)
                output_dict['xpos_loss'] = loss.item()
                accumulated_loss += loss

        if self.mlp is not None:
            mlp_output = self.mlp(encoded)
            arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
            arc_dep, arc_head = mlp_output[:, :, :arc_sz], mlp_output[:, :, arc_sz:2 * arc_sz]
            label_dep, label_head = mlp_output[:, :, 2 * arc_sz:2 * arc_sz + label_sz], \
                mlp_output[:, :, 2 * arc_sz + label_sz:]

            arc_preds = self.arc_predictor(arc_head, arc_dep)

            if adjacency_matrix is not None:
                arc_indices, arc_tags = self._transform_adjacency_matrix(adjacency_matrix)
            else:
                arc_indices, arc_tags = None, None

            # Use greedy decoding in training.
            if self.use_greedy_infer or self.training:
                head_preds = self.greedy_decoder(arc_preds, mask)  # [batch_size, seq_len]
            else:
                head_preds = self.mst_decoder(arc_preds, mask)  # [batch_size, seq_len]

            batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long,
                                       device=mask.device).unsqueeze(1)  # [batch_size, 1]
            label_head = label_head[batch_range, head_preds].contiguous()  # [batch_size, seq_len, label_mlp_size]
            label_preds = self.label_predictor(label_head, label_dep)  # [batch_size, seq_len, num_labels]

            if not self.training and self.use_intratoken_heuristics:
                # TODO: currently this part makes loss infinite
                """This piece of code prevents the model from predicting the app dependency type
                when the head is not the next token. It can only work for character-level
                models when the language does not contain any incorporation. Besides, the index
                of the app dependency type in the vocabulary must be 0."""
                shifted_index = torch.arange(1, seq_len + 1, dtype=torch.long, device=mask.device).unsqueeze(0) \
                    .repeat(batch_size, 1)
                app_masks = head_preds.ne(shifted_index)
                app_masks = app_masks.unsqueeze(2).repeat(1, 1, self.num_labels)
                app_masks[:, :, 1:] = 0
                label_preds = label_preds.masked_fill(app_masks, -float('inf'))

            output_dict.update({
                'arc_preds': arc_preds,
                'head_preds': head_preds,
                'label_preds': label_preds
            })

            if arc_indices is not None and arc_tags is not None:
                loss = self.arc_loss(arc_preds, label_preds, arc_indices, arc_tags, flipped_mask)
                output_dict['dependency_loss'] = loss.item()
                accumulated_loss += loss

        output_dict['loss'] = accumulated_loss

        return output_dict
