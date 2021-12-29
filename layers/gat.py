import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj,sparse):
        input=torch.squeeze(input)
        adj=torch.squeeze(adj.to_dense())

        h = torch.mm(input, self.W)

        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #        attention = F.dropout(attention, self.nd_dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            #            return F.elu(h_prime)
            return torch.unsqueeze(F.leaky_relu(h_prime),0)
        else:
            return torch.unsqueeze(h_prime,0)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

