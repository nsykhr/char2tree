import torch
import torch.nn as nn

from modules.utils import initial_parameter


class ArcBiaffine(nn.Module):
    """
    Biaffine Dependency Parser
    """
    def __init__(self, hidden_size, bias=True):
        super(ArcBiaffine, self).__init__()
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.has_bias = bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        initial_parameter(self)

    def forward(self, head, dep):
        """
        s_ij = dep * U * head_T + bias * head_T
        :param head: arc-head tensor [batch_size, length, hidden_size]
        :param dep: arc-dependent tensor [batch_size, length, hidden_size]
        :return output: tensor [batch_size, length, length]
        """
        output = dep.matmul(self.U)
        output = output.bmm(head.transpose(-1, -2))
        if self.has_bias:
            output = output + head.matmul(self.bias).unsqueeze(1)
        return output


class LabelBilinear(nn.Module):
    def __init__(self, in1_features, in2_features, num_label, bias=True):
        super(LabelBilinear, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label, bias=bias)
        self.lin = nn.Linear(in1_features + in2_features, num_label, bias=False)

    def forward(self, x1, x2):
        output = self.bilinear(x1, x2)
        output = output + self.lin(torch.cat([x1, x2], dim=2))
        return output
