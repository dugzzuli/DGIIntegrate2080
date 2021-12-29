import torch

from models.DGDEC import target_distribution
from utils.evaluation import eva
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
from models import LogReg
import pickle as pkl
from tqdm import tqdm
from torch.nn import Parameter
from sklearn.cluster import KMeans
import torch.nn.functional as F

#将每个视角的和综合一致性的概率P加起来，进行反向监督。


class DMGIDEC_DFCN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):

        
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        y = torch.argmax(self.labels[0], dim=1).cpu().numpy()
        idx = np.random.permutation(self.args.nb_nodes)
        shuf = [feature[:, idx, :] for feature in features]
        shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

        with torch.no_grad():
            _, _, hidden,_ = model(features, adj, shuf, self.args.sparse, None, None, None)  # 解码784，编码10

        hidden = torch.squeeze(hidden)

        kmeans = KMeans(n_clusters=self.args.nb_classes, n_init=20)  # n_init：用不同的聚类中心初始化值运行算法的次数
        y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())  # 训练并直接预测
        eva(y, y_pred, 'kmeans')
        y_pred_last = y_pred
        initCenter=kmeans.cluster_centers_
        model.cluster_layer.data = torch.tensor(initCenter).cuda()

        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        best = 1e9
        b_xent = nn.BCEWithLogitsLoss()

        iters=range(self.args.nb_epochs)

        original_acc=-1
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
            if epoch % self.args.T == 0:
                with torch.no_grad():
                    _, tmp_q, zTemp,temp_q_list = model(features, adj, shuf, self.args.sparse, None, None, None)  # 解码784，编码10
                tmp_q = tmp_q.data

                p = target_distribution(tmp_q)
                for each_q in temp_q_list:

                    p = p+ target_distribution(each_q.data)

                p=p/(self.args.nb_classes)

                res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                try:
                    eva(y, res1, str(epoch) + 'Q',Flag=True)
                except Exception:
                    print("erroir")
                finally:
                    pass
                res3 = p.data.cpu().numpy().argmax(1)  # P
                try:
                    eva(y, res3, str(epoch) + 'P',Flag=True)
                except Exception:
                    print("erroir")
                finally:
                    pass
                delta_label = np.sum(res3 != y_pred_last).astype(
                    np.float32) / res3.shape[0]

                y_pred_last = res3

                kmeans = KMeans(n_clusters=self.args.nb_classes, n_init=20)  # n_init：用不同的聚类中心初始化值运行算法的次数
                res_k = kmeans.fit_predict(zTemp.data.cpu().numpy())  # 训练并直接预测
                try:
                    acck,nmik,arik,_,_,_=eva(y, res_k, str(epoch) + 'k',Flag=True)
                except Exception:
                    print("erroir")
                finally:
                    pass

                if acck > original_acc:
                    original_acc = acck
                    torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.embedder))

                if epoch > 0 and delta_label < self.args.tol:
                    print('delta_label {:.4f}'.format(delta_label), '< tol',
                          self.args.tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break



            result, q, zTemp,q_list_eachview = model(features, adj, shuf, self.args.sparse, None, None, None)


            loss_kl = F.kl_div(q.log()
                               , p, reduction='batchmean')  # 第一个参数传入的是一个对数概率矩阵，第二个参数传入的是概率矩阵。
            for temp_q in q_list_eachview:
                loss_kl =loss_kl+ F.kl_div(temp_q.log()
                                   , p, reduction='batchmean')  # 第一个

            logits = result['logits']
            for view_idx, logit in enumerate(logits):
                if xent_loss is None:
                    xent_loss = b_xent(logit, lbl)
                else:
                    xent_loss += b_xent(logit, lbl)

            loss = xent_loss+loss_kl*self.args.lambdapra
            reg_loss = result['reg_loss']
            loss += self.args.reg_coef * reg_loss
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait =+ 1
            if cnt_wait == self.args.patience:
                break
            loss.backward()
            optimiser.step()

        model.load_state_dict(torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.embedder)))

        with torch.no_grad():
            _, tmp_q, zTemp,_ = model(features, adj, shuf, self.args.sparse, None, None, None)  # 解码784，编码10

            #使用k进行聚类，不适用tq
            # tmp_q = tmp_q.data
            # y_pred = tmp_q.cpu().numpy().argmax(1)
            # train_lbls = torch.argmax(self.labels[0, :], dim=1)
            # train_lbls = np.array(train_lbls.cpu())
            # nmi,acc,ari,stdacc,stdnmi,stdari=run_kmeans_yypred(y_pred, train_lbls)

            kmeans = KMeans(n_clusters=self.args.nb_classes, n_init=20)  # n_init：用不同的聚类中心初始化值运行算法的次数
            res_k = kmeans.fit_predict(zTemp.data.cpu().numpy())  # 训练并直接预测
            try:
                acc, nmi, ari, _, _, _ = eva(y, res_k, str(-1) + 'k', Flag=True)
            except Exception:
                print("erroir")
            finally:
                pass


            return nmi,acc,ari,0,0,0,""

class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.dmgi=DMGI(self.args)
        self.pretrain_path = self.args.pretrain_path
        self.pretrain()
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.args.nb_classes, self.args.hid_units))  # 对权重进行初始化
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def pretrain(self, path=''):
        # load pretrain weights
        # --ae.load_state_dict：将torch.load加载出来的数据加载到net中
        # --torch.load：加载训练好的模型
        self.dmgi.load_state_dict(torch.load(self.pretrain_path),False)
        print('load pretrained ae from ', self.pretrain_path)

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):

        result= self.dmgi(feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2)
        H=self.dmgi.H
        h_list=result['h_each']

        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(torch.squeeze(H).unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q_list=[]
        for h_single in h_list:
            temp_q = 1.0 / (1.0 + torch.sum(
                torch.pow(torch.squeeze(h_single).unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
            temp_q = temp_q.pow((1 + 1.0) / 2.0)
            temp_q = (temp_q.t() / torch.sum(temp_q, 1)).t()
            q_list.append(temp_q)



        return result, q, torch.squeeze(H),q_list

class DMGI(nn.Module):
    def __init__(self, args):
        super(DMGI, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(hid, args.hid_units, args.activation, args.drop_prob, args.isBias) for _, hid in
                                  zip(range(args.nb_graphs), self.args.dims)])

        self.disc = Discriminator(args.hid_units)
        # self.disc=nn.ModuleList([Discriminator(args.hid_units) for _ in range(self.args.View_num)])

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
        h_1_all = [];
        h_2_all = [];
        c_all = [];
        logits = []
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

        result['h_each'] = h_1_all
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


        # consensus regularizer
        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss

        result['reg_loss'] = reg_loss


        # self.h_1_all=h_1_all

        return result
