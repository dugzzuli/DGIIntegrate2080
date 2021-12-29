import torch

from utils.wkmeans import WKMeans

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise,adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")

def evaluate(embeds, idx_train, labels, device, isTest=True):

    nb_classes = labels.shape[2]

    train_embs = embeds[0, idx_train]


    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    train_embs = np.array(train_embs.cpu())
    train_lbls = np.array(train_lbls.cpu())
    nmi,acc,ari,stdacc,stdnmi,stdari,clu_distance=run_kmeans(train_embs, train_lbls, nb_classes)
    return nmi,acc,ari,stdacc,stdnmi,stdari

def acc_val(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    ACM_list = []
    ARI_list=[]
    for i in range(1):
        estimator.fit(x)

        y_pred = estimator.predict(x)
        acc = acc_val(np.array(y), np.array(y_pred))
        ACM_list.append(acc)
        s1 = normalized_mutual_info_score(y, y_pred)
        ari=adjusted_rand_score(y, y_pred)
        ARI_list.append(ari)
        NMI_list.append(s1)

    nmiscore = sum(NMI_list) / len(NMI_list)
    acc = sum(ACM_list) / len(ACM_list)
    ari=sum(ARI_list)/len(ARI_list)

    return nmiscore,acc,ari,np.std(ACM_list),np.std(NMI_list),np.std(ARI_list),estimator.inertia_

def run_kmeans_yypred(y_pred, y):

    NMI_list = []
    ACM_list = []
    ARI_list=[]
    for i in range(1):
        acc = acc_val(np.array(y), np.array(y_pred))
        ACM_list.append(acc)
        s1 = normalized_mutual_info_score(y, y_pred)
        ari=adjusted_rand_score(y, y_pred)
        ARI_list.append(ari)
        NMI_list.append(s1)

    nmiscore = sum(NMI_list) / len(NMI_list)
    acc = sum(ACM_list) / len(ACM_list)
    ari=sum(ARI_list)/len(ARI_list)

    return nmiscore,acc,ari,np.std(ACM_list),np.std(NMI_list),np.std(ARI_list)
