# Code based on https://github.com/PetarV-/DGI/blob/master/models/dgi.py
import torch

from models import DGI
from utils.evaluation import eva

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator
import numpy as np
np.random.seed(0)
from evaluate import evaluate
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


class DGDEC(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):

        features_lst = [feature.to(self.args.device) for feature in self.features]
        adj_lst = [adj_.to(self.args.device) for adj_ in self.adj]
        y = torch.argmax(self.labels[0], dim=1).cpu().numpy()

        final_embeds = []
        final_labels = []

        for m_idx, (features, adj) in enumerate(zip(features_lst, adj_lst)):

            self.args.m_idx=m_idx
            print("- Training on {} View".format(m_idx+1))
            model = modeler(self.args).to(self.args.device)

            idx = np.random.permutation(self.args.nb_nodes)
            shuf_fts = features[:, idx, :].to(self.args.device)

            # if(initCenterIs):

            with torch.no_grad():
                _, _, hidden = model(features, shuf_fts, adj, self.args.sparse, None, None, None)  # 解码784，编码10


            hidden = torch.squeeze(hidden)
            kmeans = KMeans(n_clusters=self.args.nb_classes, n_init=20)  # n_init：用不同的聚类中心初始化值运行算法的次数
            y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())  # 训练并直接预测

            eva(y, y_pred, 'kmeans')

            y_pred_last = y_pred
            initCenter=kmeans.cluster_centers_
            initCenterIs=False

            model.cluster_layer.data = torch.tensor(initCenter).cuda()  # kmeans.cluster_centers_：返回中心的坐标

            optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
            cnt_wait = 0; best = 1e9
            b_xent = nn.BCEWithLogitsLoss()

            accMax = -1

            for epoch in range(self.args.nb_epochs):
                model.train()
                optimiser.zero_grad()

                idx = np.random.permutation(self.args.nb_nodes)
                shuf_fts = features[:, idx, :].to(self.args.device)

                lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
                lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                lbl = lbl.to(self.args.device)

                # update target distribution p
                if epoch % self.args.T == 0:
                    with torch.no_grad():
                        _, tmp_q, zTemp = model(features, shuf_fts, adj, self.args.sparse, None, None, None)

                    tmp_q = tmp_q.data
                    p = target_distribution(tmp_q)

                    res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                    eva(y, res1, str(epoch) + 'Q')

                    res3 = p.data.cpu().numpy().argmax(1)  # P
                    acc,nmi,ari,_,_,_=eva(y, res3, str(epoch) + 'P')

                    delta_label = np.sum(res3 != y_pred_last).astype(
                        np.float32) / res3.shape[0]
                    y_pred_last = res3

                    if epoch > 0 and delta_label < self.args.tol:
                        print('delta_label {:.4f}'.format(delta_label), '< tol',
                              self.args.tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                    kmeans = KMeans(n_clusters=self.args.nb_classes, n_init=20)  # n_init：用不同的聚类中心初始化值运行算法的次数
                    res_k = kmeans.fit_predict(zTemp.data.cpu().numpy())  # 训练并直接预测
                    eva(y, res_k, str(epoch) + 'k')

                    if (accMax < acc):
                        model.eval()
                        accMax = acc

                        torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, m_idx+1))


                logits, q, zTemp = model(features, shuf_fts, adj, self.args.sparse, None, None, None)
                loss_kl = F.kl_div(q.log(), p, reduction='batchmean')  # 第一个参数传入的是一个对数概率矩阵，第二个参数传入的是概率矩阵。
                loss_xent = b_xent(logits, lbl)
                loss = 0.1 * loss_kl + loss_xent
                if loss < best:
                    best = loss
                    cnt_wait = 0

                else:
                    cnt_wait += 1

                if cnt_wait == self.args.patience:
                    break

                loss.backward()
                optimiser.step()


            model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, m_idx+1)))

            # Evaluation
            embeds, _ ,cluq = model.embed(features, adj, self.args.sparse,None)

            p = target_distribution(cluq)
            res4 = p.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, stdacc, stdnmi, stdari =eva(y, res4, str(epoch) + 'P')
            retxt = "metapath={} loss:{} epoch:{} acc:{} nmi:{}  curepoch:{}".format(m_idx+1, loss.item(), epoch, acc,
                                                                                           nmi, epoch)
            print(retxt)
            f.write(retxt)
            f.write("\n")
            final_labels.append(res4)
            final_embeds.append(embeds)
            f.flush()

        # embeds = torch.mean(torch.cat(final_embeds), 0).unsqueeze(0)
        # print("- Integrated")
        # nmi, acc, ari, stdacc, stdnmi, stdari = evaluate(embeds, self.idx_train, self.labels, self.args.device)
        # retxt1 = "alll_mean=loss:{} epoch:{} acc:{} nmi:{}  curepoch:{}".format(loss.item(), epoch, acc, nmi, epoch)
        # print(retxt1)
        #
        # f.write(retxt)
        # f.write("\n")
        #
        # embeds = torch.cat(final_embeds, 2)
        # print("- Integrated")
        # nmi, acc, ari, stdacc, stdnmi, stdari = evaluate(embeds, self.idx_train, self.labels, self.args.device)
        # retxt2 = "alll_cat=loss:{} epoch:{} acc:{} nmi:{}  curepoch:{}".format(loss.item(), epoch, acc, nmi, epoch)
        # print(retxt2)

        return nmi, acc, ari, stdacc, stdnmi, stdari



class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.dgi=DGI(self.args)
        self.pretrain_path = self.args.pretrain_path

        self.pretrain()
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.args.num_cluster, self.args.n_h))  # 对权重进行初始化
        torch.nn.init.xavier_normal_(self.cluster_layer.data)



    def pretrain(self, path=''):

        # load pretrain weights
        # --ae.load_state_dict：将torch.load加载出来的数据加载到net中
        # --torch.load：加载训练好的模型
        self.dgi.load_state_dict(torch.load(self.pretrain_path[self.args.m_idx]))
        print('load pretrained ae from ', self.pretrain_path[self.args.m_idx])

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):

        ret, h_1 = self.dgi(seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)

        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(torch.squeeze(h_1).unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return ret, q, torch.squeeze(h_1)

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1,c= self.dgi.embed(seq, adj, sparse)

        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(torch.squeeze(h_1).unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return h_1.detach(), c.detach(),q.detach()

from torch.nn import Parameter
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class DGI(nn.Module):
    def __init__(self, args):
        super(DGI, self).__init__()
        self.args = args
        self.gcn = GCN(args.ft_size[args.m_idx], args.hid_units, args.activation, args.drop_prob, args.isBias)

        # one discriminator
        self.disc = Discriminator(args.hid_units)
        self.readout_func = self.args.readout_func

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.readout_func(h_1)  # equation 9
        c = self.args.readout_act_func(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret,h_1

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)

        c = self.readout_func(h_1)  # positive summary vector
        c = self.args.readout_act_func(c)  # equation 9

        return h_1, c

