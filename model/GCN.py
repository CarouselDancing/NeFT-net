import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer with learnable adjacency (attention) matrix.
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # Feature transformation
        self.att = Parameter(torch.FloatTensor(node_n, node_n))  # Learnable adjacency matrix

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)  # Linear transformation of node features
        output = torch.matmul(self.att, support)  # Apply adjacency matrix (attention)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        A residual block with two GCN layers.
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

    def forward(self, x):
        # First GCN layer
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)  # Batch normalization
        y = self.activation(y)
        y = self.dropout(y)

        # Second GCN layer
        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        y = self.dropout(y)

        return y + x  # Residual connection

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        GCN with multiple stages of residual blocks.
        :param input_feature: Number of input features per node.
        :param hidden_feature: Number of hidden features in the GCN layers.
        :param p_dropout: Dropout probability.
        :param num_stage: Number of residual blocks (GC_Block) to stack.
        :param node_n: Number of nodes in the graph.
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        # First Graph Convolution Layer
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        # Create Residual Blocks
        self.gcbs = nn.ModuleList([GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n) for _ in range(num_stage)])

        # Final Graph Convolution Layer
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

    def forward(self, x, is_out_resi=True):
        # Initial GCN layer
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)  # Apply batch normalization
        y = self.activation(y)
        y = self.dropout(y)

        # Pass through residual GCN blocks
        for gc_block in self.gcbs:
            y = gc_block(y)

        # Final GCN layer
        y = self.gc7(y)

        # Apply residual connection if enabled
        if is_out_resi:
            y = y + x

        return y
