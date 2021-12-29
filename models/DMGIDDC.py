import torch

from utils.utils import mkdir

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention, get_fusion_module
import numpy as np
np.random.seed(0)
from evaluate import evaluate

from tqdm import tqdm
"""
We try to determine the different impact for clustering with different loss combination.
"""
from abc import ABC

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.evaluation import eva

class BaseLoss:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DMGIDDC(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        print(model)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.1, last_epoch=-1)

        best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        cls_criterion = DDCLoss(self.args.nb_classes, device=self.args.device)
        iters=tqdm(range(self.args.nb_epochs))
        accMax=-1

        nmiMax = -1
        ariMax=-1
        curepoch=-1

        retxt=""
        if(self.args.Fine):
            for epoch in iters:
                model.train()
                xent_loss = None
                optimiser.zero_grad()
                idx = np.random.permutation(self.args.nb_nodes)

                shuf = [feature[:, idx, :] for feature in features]
                shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

                lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
                lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

                result = model(features, adj, shuf, self.args.sparse, None, None, None)
                logits = result['logits']

                for view_idx, logit in enumerate(logits):
                    if xent_loss is None:
                        xent_loss = b_xent(logit, lbl)
                    else:
                        xent_loss += b_xent(logit, lbl)
                loss = xent_loss

                y, h=result["y"],result["h"]
                clustering_loss = cls_criterion(y, h)

                # loss += self.args.reg_coef * reg_loss
                loss+=self.args.cls_reg*clustering_loss
                if loss < best:
                    best = loss
                    cnt_wait = 0

                else:
                    cnt_wait =+ 1
                if cnt_wait == self.args.patience:
                    break
                loss.backward()
                optimiser.step()

                scheduler.step()

                # Evaluation
                if(epoch%10)==0:
                    # print(loss)
                    # model.eval()
                    with torch.no_grad():
                        temp_result=model(features, adj, shuf, self.args.sparse, None, None, None)
                        y = torch.argmax(self.labels[0], dim=1).cpu().numpy()
                        train_lbls = np.array(y)

                        y_pred=temp_result["y"].detach().cpu().max(1)[1]
                        try:
                            acc,nmi,ari,stdacc,stdnmi,stdari=eva(np.array(y_pred),train_lbls, str(epoch) + 'Q',Flag=False)
                        except:
                            pass
                        else:
                            pass
                    if(accMax<acc):
                        accMax=acc
                        nmiMax=nmi
                        ariMax=ari
                        curepoch=epoch
                        savePath = "saved_model/{}/".format(self.args.dataset)
                        mkdir(savePath)
                        # torch.save(model.state_dict(),
                        #            savePath + 'best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,
                        #                                                  self.args.isMeanOrCat))

                    retxt="loss:{} epoch:{} acc:{} nmi:{} accMax:{} nmiMax:{} ariMax:{} curepoch:{}".format(loss.item(),epoch,acc,nmi,accMax,nmiMax,ariMax,curepoch)
                    iters.set_description(retxt)

        # model.load_state_dict(torch.load('saved_model/{}/best_{}_{}_{}.pkl'.format(self.args.dataset,self.args.dataset, self.args.embedder,self.args.isMeanOrCat)),False)

        # with torch.no_grad():
        #     temp_result=model(features, adj, shuf, self.args.sparse, None, None, None)
        #     y = torch.argmax(self.labels[0], dim=1).cpu().numpy()
        #     train_lbls = np.array(y)

        #     y_pred=temp_result["y"].detach().cpu().max(1)[1]
        #     nmi,acc,ari,stdacc,stdnmi,stdari=eva(train_lbls,np.array(y_pred), str(epoch) + 'Q',Flag=True)

        return nmiMax,accMax,ariMax,stdacc,stdnmi,stdari,retxt


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(hid, args.hid_units, args.activation, args.drop_prob, args.isBias) for _,hid in zip(range(args.nb_graphs),self.args.dims)])

        self.disc=Discriminator(args.hid_units)
        
        self.fusion = get_fusion_module("weighted_mean",self.args.View_num, [args.hid_units for _ in range(self.args.View_num) ])

        self.clustering_module = DDCModule(self.args.hid_units, self.args.cluster_hidden_dim, self.args.nb_classes)
        


        if(self.args.isMeanOrCat=='Mean'):
            self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        else:
            self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units * self.args.View_num))
        self.readout_func = self.args.readout_func
        
        

        self.init_weight()
        self.apply(self.weights_init('xavier'))

    def init_weight(self):
        nn.init.xavier_normal_(self.H)
    
    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    import math
                    nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        return init_fun

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in range(self.args.nb_graphs):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)


            # how to readout positive summary vector
            c = self.readout_func(h_1)
            c = self.args.readout_act_func(c)  # equation 9
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)


            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)

        result['logits'] = logits


        result["h_1_all"] = h_1_all

        

        h_1_all = [ torch.squeeze(each) for each in h_1_all]
        h_2_all = [ torch.squeeze(each) for each in h_2_all]

        h_1_all=self.fusion(h_1_all)
        h_2_all =self.fusion(h_2_all)

        # consensus regularizer
        # pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        # neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        # reg_loss = pos_reg_loss - neg_reg_loss
        # # reg_loss = pos_reg_loss
        # # reg_loss=reg_loss if reg_loss >0 else 10
        # result['reg_loss'] = reg_loss



        y, h = self.clustering_module(h_1_all)

        result["y"]=y
        result["h"]=h


        return result

class DDCModule(nn.Module):

    def __init__(self, in_features, hidden_dim, num_cluster):
        super(DDCModule, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim, momentum=0.1)
        )

        self.clustering_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_cluster),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        y = self.clustering_layer(h)
        return y, h

class DDCLoss(BaseLoss, ABC):
    """
    Michael Kampffmeyer et al. "Deep divergence-based approach to clustering"
    """

    def __init__(self, num_cluster, epsilon=1e-9, rel_sigma=0.15, device='cpu'):
        """

        :param epsilon:
        :param rel_sigma: Gaussian kernel bandwidth
        """
        super(DDCLoss, self).__init__()
        self.epsilon = epsilon
        self.rel_sigma = rel_sigma
        self.device = device
        self.num_cluster = num_cluster

    def __call__(self, logist, hidden):
        hidden_kernel = self._calc_hidden_kernel(hidden)
        l1_loss = self._l1_loss(logist, hidden_kernel, self.num_cluster)
        l2_loss = self._l2_loss(logist)
        l3_loss = self._l3_loss(logist, hidden_kernel, self.num_cluster)
        return l1_loss + l2_loss + l3_loss

    def _l1_loss(self, logist, hidden_kernel, num_cluster):
        return self._d_cs(logist, hidden_kernel, num_cluster)

    def _l2_loss(self, logist):
        n = logist.size(0)
        return 2 / (n * (n - 1)) * self._triu(logist @ torch.t(logist))

    def _l3_loss(self, logist, hidden_kernel, num_cluster):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(num_cluster, device=self.device)
        m = torch.exp(-self._cdist(logist, self.eye))
        return self._d_cs(m, hidden_kernel, num_cluster)

    def _triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _calc_hidden_kernel(self, x):
        return self._kernel_from_distance_matrix(self._cdist(x, x), self.epsilon)

    def _d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.

        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom, eps=self.epsilon)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.epsilon ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self._triu(nom / torch.sqrt(dnom_squared))
        return d

    def _atleast_epsilon(self, X, eps):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def _cdist(self, X, Y):
        """
        Pairwise distance between rows of X and rows of Y.

        :param X: First input matrix
        :type X: torch.Tensor
        :param Y: Second input matrix
        :type Y: torch.Tensor
        :return: Matrix containing pairwise distances between rows of X and rows of Y
        :rtype: torch.Tensor
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def _kernel_from_distance_matrix(self, dist, min_sigma):
        """
        Compute a Gaussian kernel matrix from a distance matrix.

        :param dist: Disatance matrix
        :type dist: torch.Tensor
        :param min_sigma: Minimum value for sigma. For numerical stability.
        :type min_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k
