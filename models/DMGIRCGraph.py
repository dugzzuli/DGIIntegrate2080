import torch

from utils import process
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
from evaluate import evaluate
from models import LogReg
import pickle as pkl
from tqdm import tqdm
from utils.calcu_graph import construct_graph,load_graph
class DMGIRCGraph(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0;
        best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        iters=tqdm(range(self.args.nb_epochs))
        accMax=-1
        minclu_distance=1000
        nmiMax = -1
        ariMax=-1
        curepoch=-1
        min_curepoh=-1
        showResults = {}
        retxt=""
        if(self.args.Fine):
            for epoch in iters:

                model.train()
                xent_loss = None
                model.train()
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

                reg_loss = result['reg_loss']
                loss += self.args.reg_coef * reg_loss



                if loss < best:
                    best = loss
                    cnt_wait = 0

                    # torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.embedder))
                else:
                    cnt_wait =+ 1

                if cnt_wait == self.args.patience:
                    break

                loss.backward()
                optimiser.step()
                # Evaluation
                if((epoch+1)%5)==0:
                    # print(loss)
                    model.eval()

                    nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(model.H.data.detach(), self.idx_train, self.labels, self.args.device)

                    if ((epoch+1) % 200) == 0:
                        print("更新W")
                        import scipy.sparse as sp

                        genePath = "generateW/{}/".format(self.args.dataset)
                        geW = genePath + 'W.txt'

                        construct_graph(model.H.data.detach(), geW, graph=self.labels)

                        label_file = './Database/' + self.args.dataset + '/group.txt'
                        tempW=load_graph(label_file,geW)
                        tempadj_1 = [sp.csr_matrix(tempW + np.eye(self.args.nb_nodes) * self.args.sc) for _ in range(self.args.nb_graphs)]

                        tempadj_2 = [process.normalize_adj(adj_) for adj_ in tempadj_1]
                        tempadj_3 = [process.sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in tempadj_2]
                        adj = [adj_.to(self.args.device) for adj_ in tempadj_3]






                    if(accMax<acc):
                        accMax=acc
                        nmiMax=nmi
                        ariMax=ari
                        curepoch=epoch
                        savePath="saved_model/{}/".format(self.args.dataset)
                        mkdir(savePath)
                        torch.save(model.state_dict(),
                                   savePath+'best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,self.args.isMeanOrCat))



                    retxt="loss:{} epoch:{} acc:{} nmi:{} accMax:{} nmiMax:{} ariMax:{} curepoch:{}".format(loss.item(),epoch,acc,nmi,accMax,nmiMax,ariMax,curepoch)
                    iters.set_description(retxt)

                    showResults["acc"]=acc
                    showResults["accMax"] = accMax
                    if self.args.vis is not None:
                        self.args.vis.plot_many_stack(showResults)
                        self.args.vis.plot_many_stack({"loss:": loss.item()})
                    # f.write("loss:{} epoch:{} acc:{} nmi:{} accMax:{} nmiMax:{} curepoch:{}".format(loss.item(),epoch,acc,nmi,accMax,nmiMax,curepoch))
                    # f.write('\n')
                    # print()
        # args.pretrain_path = "./final_model/best_Reuters_DMGI_Mean.pkl"
        model.load_state_dict(torch.load('saved_model/{}/best_{}_{}_{}.pkl'.format(self.args.dataset,self.args.dataset, self.args.embedder,self.args.isMeanOrCat)),False)
        # model.load_state_dict(torch.load("./final_model/best_Reuters_DMGI_Mean.pkl"),False)
        model.eval()

        nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(model.H.data.detach(), self.idx_train, self.labels, self.args.device)



        return nmi,acc,ari,stdacc,stdnmi,stdari,retxt


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(hid, args.hid_units, args.activation, args.drop_prob, args.isBias) for _,hid in zip(range(args.nb_graphs),self.args.dims)])

        self.disc=Discriminator(args.hid_units)
        # self.disc=nn.ModuleList([Discriminator(args.hid_units) for _ in range(self.args.View_num)])


        if(self.args.isMeanOrCat=='Mean'):
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
            h_1_all_lst = []; h_2_all_lst = []; c_all_lst = []

            for h_idx in range(self.args.nheads):
                h_1_all_, h_2_all_, c_all_ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_); h_2_all_lst.append(h_2_all_); c_all_lst.append(c_all_)

            if (self.args.isMeanOrCat == 'Mean'):
                h_1_all = torch.mean(torch.cat(h_1_all_lst), 0).unsqueeze(0)
                h_2_all = torch.mean(torch.cat(h_2_all_lst), 0).unsqueeze(0)
            else:
                h_1_all = torch.cat(h_1_all_lst,2).squeeze().unsqueeze(0)
                h_2_all = torch.cat(h_2_all_lst,2).squeeze().unsqueeze(0)

        else:
            # print("no using attention")
            
            if (self.args.isMeanOrCat == 'Mean'):
                h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
                h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)
            else:
                h_1_all = torch.cat(h_1_all,2).squeeze().unsqueeze(0)
                h_2_all = torch.cat(h_2_all,2).squeeze().unsqueeze(0)

        result["h_1_all"] = h_1_all
        # consensus regularizer
        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        # reg_loss = pos_reg_loss
        # reg_loss=reg_loss if reg_loss >0 else 10
        result['reg_loss'] = reg_loss

        # self.h_1_all=h_1_all


        return result
