# Code based on https://github.com/PetarV-/DGI/blob/master/models/dgi.py
import torch

from layers.AutoEncoder import Encoder, Decoder

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

class AE(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):
        features_lst = [feature.to(self.args.device) for feature in self.features]
        adj_lst = [adj_.to(self.args.device) for adj_ in self.adj]

        final_embeds = []
        for m_idx, (features, adj) in enumerate(zip(features_lst, adj_lst)):
            metapath = m_idx+1
            print("- Training on {}".format(metapath))
            self.args.m_idx=m_idx
            model = modeler(self.args).to(self.args.device)

            optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
            cnt_wait = 0; best = 1e9
            b_xent = nn.MSELoss()
            for epoch in range(self.args.nb_epochs):
                model.train()
                optimiser.zero_grad()
                x_pro, z = model(features)
                loss = b_xent(x_pro, features)
                if loss < best:
                    best = loss
                    cnt_wait = 0
                    torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, metapath))
                else:
                    cnt_wait += 1

                if cnt_wait == self.args.patience:
                    break

                loss.backward()
                optimiser.step()

            model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, metapath)))

            # Evaluation
            embeds = model.embed(features)
            nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(embeds, self.idx_train, self.labels, self.args.device)
            retxt = "metapath={} loss:{} epoch:{} acc:{} nmi:{}  curepoch:{}".format(metapath,loss.item(),epoch, acc, nmi,epoch)
            print(retxt)
            f.write(retxt)
            f.write("\n")
            final_embeds.append(embeds)

        embeds = torch.mean(torch.cat(final_embeds), 0).unsqueeze(0)
        print("- Integrated")
        nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(embeds, self.idx_train, self.labels, self.args.device)
        retxt1 = "alll_mean=loss:{} epoch:{} acc:{} nmi:{}  curepoch:{}".format(loss.item(), epoch, acc, nmi, epoch)
        print(retxt1)

        f.write(retxt1)
        f.write("\n")

        embeds = torch.cat(final_embeds,2)
        print("- Integrated")
        nmi, acc, ari, stdacc, stdnmi, stdari = evaluate(embeds, self.idx_train, self.labels, self.args.device)
        retxt2 = "alll_cat=loss:{} epoch:{} acc:{} nmi:{}  curepoch:{}".format(loss.item(), epoch, acc, nmi, epoch)
        print(retxt2)

        return nmi,acc,ari,stdacc,stdnmi,stdari,retxt2


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args

        self.encoder = Encoder(args.ft_size[args.m_idx], self.args.inter_dims, self.args.nb_classes)
        self.decoder = Decoder(args.ft_size[args.m_idx], self.args.inter_dims, self.args.nb_classes)


    def forward(self, x):
        z = self.encoder(x)
        self.z = z
        x_pro = self.decoder(z)
        return x_pro, z

    # Detach the return variables
    def embed(self, x):
        z = self.encoder(x)

        return z.detach()
