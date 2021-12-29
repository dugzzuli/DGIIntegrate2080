import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn

class InterDiscriminator(nn.Module):
    def __init__(self, n_h, ft_size):
        super().__init__()
        self.f_k_bilinear_e = nn.Bilinear(n_h, n_h, 1)
        self.f_k_bilinear_i = nn.Bilinear(ft_size, n_h, 1)
        self.f_k_bilinear_j = nn.Bilinear(n_h, n_h, 1)

        self.linear_c = nn.Linear(n_h, n_h)
        self.linear_f = nn.Linear(ft_size, n_h)
        self.linear_cf = nn.Linear(n_h*2, n_h)

        self.act = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, f_pl, f_mi):
        """
        :param c: global summary vector, [dim]
        :param h_pl: positive local vectors [n_nodes, dim]
        :param h_mi: negative local vectors [n_nodes, dim]
        :param f: features/attributes [n_nodes, dim_f]
        """
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        c_x_0 = self.act(c_x)

        # extrinsic
        logits_1 = torch.squeeze(self.f_k_bilinear_e(c_x_0, h_pl))
        logits_2 = torch.squeeze(self.f_k_bilinear_e(c_x_0, h_mi))
        logits_nodes = torch.stack([logits_1, logits_2], 0)

        # intrinsic
        logits_1 = torch.squeeze(self.f_k_bilinear_i(f_pl, h_pl))
        logits_2 = torch.squeeze(self.f_k_bilinear_i(f_pl, h_mi))
        logits_locs = torch.stack([logits_1, logits_2], 0)

        # joint
        c_x = self.act(c_x)
        c_x = self.linear_c(c_x)
        f_pl = self.linear_f(f_pl)
        f_mi = self.linear_f(f_mi)
        c_x = self.act(c_x)
        f_pl = self.act(f_pl)
        f_mi = self.act(f_mi)

        cs_pl = torch.cat([c_x, f_pl], dim=-1)
        cs_mi = torch.cat([c_x, f_mi], dim=-1)

        cs_pl = self.linear_cf(cs_pl)
        cs_mi = self.linear_cf(cs_mi)
        cs_pl = self.act(cs_pl)
        cs_mi = self.act(cs_mi)

        logits_1 = torch.squeeze(self.f_k_bilinear_j(cs_pl, h_pl))
        logits_2 = torch.squeeze(self.f_k_bilinear_j(cs_mi, h_pl))
        logits_cs = torch.stack([logits_1, logits_2], 0)

        return logits_nodes, logits_locs, logits_cs


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1) # c: summary vector, h_pl: positive, h_mi: negative
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), 2) # sc_1 = 1 x nb_nodes
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), 2) # sc_2 = 1 x nb_nodes

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)

        return logits