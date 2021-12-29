import numpy as np
import linecache
from Dataset.dataset import Dataset
from sklearn.metrics import pairwise_distances as pair
import random
from sklearn.preprocessing import normalize
def construct_graph(embeds,fname, method='heat',topk=15,graph=None):
    train_embs = embeds[0, :]

    features = np.array(train_embs.cpu())

    labels=graph[0, :].cpu().numpy()

    m,n=np.shape(features)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    if(topk>0):
        inds = []
        for i in range(dist.shape[0]):
            ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
            inds.append(ind)

        with open(fname, mode="w") as f:
            for i, v in enumerate(inds):
                mutual_knn = False
                for vv in v:
                    if vv == i:
                        pass
                    else:
                        f.write('{} {} {} {}\n'.format(i, vv,np.argmax(labels[i]),np.argmax(labels[vv])))
    else:
        with open(fname, mode="w") as f:
            for k in range(m):
                f.write(str(k))
                for j in range(m):
                    f.write(" " + str(dist[k][j]))
                f.write("\n")
            f.flush()

def load_graph(label_file, file):
    lines = linecache.getlines(label_file)
    lines = [line.rstrip('\n') for line in lines]
    node_map = {}
    for idx, line in enumerate(lines):
        line = line.split(' ')
        node_map[line[0]] = idx

    num_nodes = len(node_map)

    VData = np.zeros((num_nodes, num_nodes))

    lines = linecache.getlines(file)
    lines = [line.rstrip('\n') for line in lines]
    for line in lines:
        line = line.split(' ')
        idx1 = node_map[line[0]]
        idx2 = node_map[line[1]]
        VData[idx2, idx1] = 1.0
        VData[idx1, idx2] = 1.0

    return VData

