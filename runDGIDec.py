import numpy as np

from utils import process
from utils.utils import mkdir

np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
import os
import yaml


if __name__ == '__main__':

    d=['Reuters'] #['Reuters','yale_mtv','MSRCv1','3sources','small_Reuters','small_NUS','BBC','BBCSport'] # ['BBCSport','yale_mtv','MSRCv1','3sources']
    for data in d:
        config = yaml.load(open("configMain.yaml", 'r'))

        # input arguments
        parser = argparse.ArgumentParser(description='DGDEC')
        parser.add_argument('--embedder', nargs='?', default='DGDEC')
        parser.add_argument('--dataset', nargs='?', default=data)
        parser.add_argument('--View_num',default=config[data]['View_num'])
        parser.add_argument('--norm',default=config[data]['norm'])
        parser.add_argument('--nb_epochs', type=int, default=config[data]['nb_epochs'])
        parser.add_argument('--sc', type=float,default=10.0, help='GCN self connection') #config[data]['sc']
        parser.add_argument('--gpu_num', type=int, default=0)
        parser.add_argument('--drop_prob', type=float, default=0.2)
        parser.add_argument('--patience', type=int, default=100)
        parser.add_argument('--activation', nargs='?', default='leakyrelu')
        parser.add_argument('--isBias',default=False)
        parser.add_argument('--Weight', nargs='?', default=config['Weight'])
        parser.add_argument('--lr', type=float, default=0.001, help='学习率')
        parser.add_argument('--hid_units', type=int, default=512, help='低维特征维度')
        parser.add_argument('--n_h', type=int, default=512, help='低维特征维度')
        parser.add_argument('--l2_coef', type=float, default=0.01, help='l2_coef')
        parser.add_argument('--tol', type=float, default=0.0001, help='tol')
        parser.add_argument('--T', type=int, default=5, help='T')




        args, unknown = parser.parse_known_args()
        print(args)
        args.pretrain_path=["./saved_model/best_{}_{}_{}.pkl".format(args.dataset,"DGI",index+1) for index in range(args.View_num)]

        resultsDir = 'baseline/{}/{}'.format(args.embedder, args.dataset)

        mkdir(resultsDir)

        filePath = os.path.join(resultsDir, '{}_{}_{}_sc.{}.txt'.format('Y 'if args.Weight else 'N',args.dataset,config[args.dataset]['norm'],args.sc))


        with open(filePath, 'a+') as f:
            f.write("SC:{}\n".format(args.sc))
            f.flush()

            rownetworks, truefeatures_list, labels, idx_train=process.load_data_mv(args,Unified=False)

            args.rownetworks, args.truefeatures_list, args.labels, args.idx_train=rownetworks, truefeatures_list, labels, idx_train
            args.num_cluster = len(np.unique(np.argmax(labels, 1)))

            print(args)
            from models import DGDEC
            embedder = DGDEC(args)
            nmi, acc, ari, stdacc, stdnmi, stdari = embedder.training(f)
            result = "hid_units:{},lr:{},l2_coef:{},acc:{},nmi:{},Ari:{},stdnmi:{},stdacc:{},stdari:{}".format(
                args.hid_units, args.lr, args.l2_coef, acc, nmi, ari, stdacc, stdnmi, stdari)

            f.write(result)
            f.write('\n')
            f.write('\n')
            f.flush()

                