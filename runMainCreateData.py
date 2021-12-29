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

    d=['NUSWIDEOBJ'] #['3sources','Reuters','small_Reuters','small_NUS','LandUse-21','BBCSport'] #['Reuters','yale_mtv','MSRCv1','3sources','small_Reuters','small_NUS','BBC','BBCSport'，‘LandUse-21’] # ['BBCSport','yale_mtv','MSRCv1','3sources']
    atten=False
    # vis = Visualizer("env")
    vis=None
    for data in d:
        for link in ['Mean']:
            config = yaml.load(open("configMain.yaml", 'r'))
            
            # input arguments
            parser = argparse.ArgumentParser(description='DMGI')
            parser.add_argument('--embedder', nargs='?', default='DMGI')
            parser.add_argument('--dataset', nargs='?', default=data)
            parser.add_argument('--View_num',default=config[data]['View_num'])
            parser.add_argument('--norm',default=config[data]['norm'])
            parser.add_argument('--nb_epochs', type=int, default=config[data]['nb_epochs'])
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


            args, unknown = parser.parse_known_args()

            args.vis=vis
            args.Fine = True

            import scipy.io as sio

            rownetworks, truefeatures_list, labels, idx_train = process.load_data_mv(args, Unified=False)
            dataD={}
            for i,fea in enumerate(truefeatures_list):
                dataD["view_{}".format(i)]=fea
            # dataD["labels"]=np.argmax(labels,axis=1)
            mkdir("./Database/npzdata/")
            np.savez("./Database/npzdata/{}.npz".format(data),labels=np.argmax(labels,axis=1),n_views=len(truefeatures_list),
                     view_0=truefeatures_list[0].todense(),
                     view_1=truefeatures_list[1].todense(),
                     view_2=truefeatures_list[2].todense(),
                     view_3=truefeatures_list[3].todense(),
                     view_4=truefeatures_list[4].todense(),
                     )






                
                