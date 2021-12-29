import math

import torch

from utils.utils import mkdir

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention
import numpy as np
np.random.seed(0)
from evaluate import evaluate, run_kmeans_yypred

from tqdm import tqdm
# Contrastive Clustering (CC)
class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()

        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class DMGINoCSWCC(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        criterion_cluster = ClusterLoss(self.args.nb_classes, 1.0, torch.device("cuda")).to(
            torch.device("cuda"))

        iters=tqdm(range(self.args.nb_epochs))
        accMax=-1

        nmiMax = -1
        ariMax=-1
        curepoch=-1
        showResults = {}
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
                logits_all=result['logits_all']
                for view_idx, logit in enumerate(logits):
                    if xent_loss is None:
                        xent_loss = b_xent(logit, lbl)
                    else:
                        xent_loss += b_xent(logit, lbl)

                xent_loss += b_xent(logits_all, lbl)

                loss = xent_loss+criterion_cluster(result["c_i"], result["c_j"])*100

                if loss < best:
                    best = loss
                    cnt_wait = 0
                else:
                    cnt_wait =+ 1
                if cnt_wait == self.args.patience:
                    break
                loss.backward()
                optimiser.step()
                # Evaluation
                if(epoch%5)==0:
                    with torch.no_grad():
                        c = model.forward_cluster(features, adj, shuf, self.args.sparse, None, None,
                                                  None)  # 解码784，编码10

                    nmi, acc, ari, stdacc, stdnmi, stdari = run_kmeans_yypred(c.cpu().numpy(),
                                                                              np.argmax(torch.squeeze(self.labels).cpu().numpy(),axis=1))
                    if(accMax<acc):
                        accMax=acc
                        nmiMax=nmi
                        ariMax=ari
                        curepoch=epoch
                        savePath="saved_model/{}/".format(self.args.dataset)
                        mkdir(savePath)
                        torch.save(model.state_dict(),savePath+'best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,self.args.isMeanOrCat))
                    retxt="loss:{} epoch:{} acc:{} nmi:{} accMax:{} nmiMax:{} ariMax:{} curepoch:{}".format(loss.item(),epoch,acc,nmi,accMax,nmiMax,ariMax,curepoch)
                    iters.set_description(retxt)
                    showResults["acc"]=acc
                    showResults["accMax"] = accMax
                    if self.args.vis is not None:
                        self.args.vis.plot_many_stack(showResults)
                        self.args.vis.plot_many_stack({"loss:": loss.item()})

        model.load_state_dict(torch.load('saved_model/{}/best_{}_{}_{}.pkl'.format(self.args.dataset,self.args.dataset, self.args.embedder,self.args.isMeanOrCat)),False)

        model.eval()
        with torch.no_grad():
            c = model.forward_cluster(features, adj, shuf, self.args.sparse, None, None,
                           None)  # 解码784，编码10

        nmi, acc, ari, stdacc, stdnmi, stdari = run_kmeans_yypred(c.cpu().numpy(),
                                                                  np.argmax(torch.squeeze(self.labels).cpu().numpy(),
                                                                            axis=1))
        return nmi,acc,ari,stdacc,stdnmi,stdari,retxt


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(hid, args.hid_units, args.activation, args.drop_prob, args.isBias) for _,hid in zip(range(args.nb_graphs),self.args.dims)])

        self.disc=Discriminator(args.hid_units)

        self.discAll = Discriminator(args.hid_units)

        self.cluster_projector = nn.Sequential(
            nn.Linear(self.args.hid_units, self.args.hid_units),
            nn.ReLU(),
            nn.Linear(self.args.hid_units, self.args.nb_classes),
            nn.Softmax(dim=1)
        )
        if (self.args.isMeanOrCat == 'Mean'):
            self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        else:
            self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units * self.args.View_num))

        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

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

        # Attention or not
        if self.args.isAttn:
            # print("using attention")
            h_1_all_lst = [];
            h_2_all_lst = [];
            c_all_lst = []

            for h_idx in range(self.args.nheads):
                h_1_all_, h_2_all_, c_all_ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_);
                h_2_all_lst.append(h_2_all_);
                c_all_lst.append(c_all_)

            if (self.args.isMeanOrCat == 'Mean'):
                h_1_all = torch.mean(torch.cat(h_1_all_lst), 0).unsqueeze(0)
                h_2_all = torch.mean(torch.cat(h_2_all_lst), 0).unsqueeze(0)
            else:
                h_1_all = torch.cat(h_1_all_lst, 2).squeeze().unsqueeze(0)
                h_2_all = torch.cat(h_2_all_lst, 2).squeeze().unsqueeze(0)

        else:
            # print("no using attention")

            if (self.args.isMeanOrCat == 'Mean'):
                h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
                h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)
            else:
                h_1_all = torch.cat(h_1_all, 2).squeeze().unsqueeze(0)
                h_2_all = torch.cat(h_2_all, 2).squeeze().unsqueeze(0)

        result["h_1_all"] = h_1_all

        c_all = self.readout_func(h_1_all)
        c_all = self.args.readout_act_func(c_all)  # equation 9
        logit_all = self.discAll(c_all, h_1_all, h_2_all)
        result['logits_all']=logit_all

        c_i = self.cluster_projector(torch.squeeze(h_1_all))
        c_j = self.cluster_projector(torch.squeeze(h_2_all))
        result["c_i"]=c_i
        result["c_j"] = c_j

        return result

    def forward_cluster(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []
        for i in range(self.args.nb_graphs):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)
            h_1_all.append(h_1)

        if (self.args.isMeanOrCat == 'Mean'):
            h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)

        else:
            h_1_all = torch.cat(h_1_all, 2).squeeze().unsqueeze(0)

        c = self.cluster_projector(torch.squeeze(h_1_all))

        c = torch.argmax(c, dim=1)
        return c
