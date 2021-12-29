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

    d=['3sources'] #['Reuters','yale_mtv','MSRCv1','3sources','small_Reuters','small_NUS','BBC','BBCSport'] # ['BBCSport','yale_mtv','MSRCv1','3sources']
    atten=False
    for data in d:
        for link in ['Mean']:
            config = yaml.load(open("configMain.yaml", 'r'))
            
            # input arguments
            parser = argparse.ArgumentParser(description='DMGIDEC_DFCN_Single')
            parser.add_argument('--embedder', nargs='?', default='DMGIDEC_DFCN_Single')
            parser.add_argument('--dataset', nargs='?', default=data)
            parser.add_argument('--View_num',default=config[data]['View_num'])
            parser.add_argument('--norm',default=config[data]['norm'])
            parser.add_argument('--nb_epochs', type=int, default=config[data]['DEC_nb_epochs'])
            parser.add_argument('--sc', type=float,default=10.0, help='GCN self connection') #config[data]['sc']
            parser.add_argument('--gpu_num', type=int, default=0)
            parser.add_argument('--drop_prob', type=float, default=0.2)
            parser.add_argument('--patience', type=int, default=100)
            parser.add_argument('--nheads', type=int, default=1)
            parser.add_argument('--activation', nargs='?', default='leakyrelu')
            parser.add_argument('--isBias',default=False)
            parser.add_argument('--isAttn',  default=atten)
            
            parser.add_argument('--isMeanOrCat', nargs='?', default=link) #config[data]['isMeanOrCat']
            parser.add_argument('--Weight', nargs='?', default=config['Weight'])

            parser.add_argument('--lambdapra', type=float, default=0.1, help='lambdapra')
            parser.add_argument('--tol', type=float, default=-1, help='reg_coef')

            parser.add_argument('--lr', type=float, default=0.001, help='学习率')
            parser.add_argument('--hid_units', type=int, default=128, help='低维特征维度')
            parser.add_argument('--l2_coef', type=float, default=0.00001, help='l2_coef')
            parser.add_argument('--reg_coef', type=float, default=0.00001, help='reg_coef')
            parser.add_argument('--T', type=int, default=3, help='更新迭代')
            
                
            args, unknown = parser.parse_known_args()
                        
            print(args)
            args.pretrain_path = "./saved_model/{}/best_{}_{}_{}.pkl".format(args.dataset,args.dataset,"DMGI",args.isMeanOrCat)
            # args.pretrain_path = "./final_model/best_Reuters_DMGI_Mean1080ti.pkl"

            resultsDir = 'baseline/{}/{}/{}'.format(args.isMeanOrCat,args.embedder,args.dataset)
            mkdir(resultsDir)
            
            filePath = os.path.join(resultsDir, '{}_{}_{}_{}_sc.{}.txt'.format('Y 'if args.Weight else 'N',args.dataset,args.isMeanOrCat,config[args.dataset]['norm'],args.sc))

            
            with open(filePath, 'a+') as f:
                f.write("SC:{}\n".format(args.sc))
                f.flush()
                
                rownetworks, truefeatures_list, labels, idx_train=process.load_data_mv(args,Unified=False)

                args.rownetworks, args.truefeatures_list, args.labels, args.idx_train=rownetworks, truefeatures_list, labels, idx_train 
                
                print(args)
                from models import DMGIDEC_DFCN_Single
                embedder = DMGIDEC_DFCN_Single(args)
                nmi, acc, ari, stdacc, stdnmi, stdari, retxt = embedder.training(f)

                result = "hid_units:{},lr:{},l2_coef:{},reg_coef:{},acc:{},nmi:{},Ari:{},stdnmi:{},stdacc:{},stdari:{}".format(
                    args.hid_units, args.lr, args.l2_coef, args.reg_coef, acc, nmi, ari, stdacc, stdnmi, stdari)

                f.write(retxt)
                f.write('\n')
                f.write(result)
                f.write('\n')
                f.write('\n')
                f.flush()
                
                