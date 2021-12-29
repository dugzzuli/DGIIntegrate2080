import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
import torch
from utils.wkmeans import WKMeans


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




def small_trick(y_test, y_pred):
    y_pred_new = np.zeros(y_pred.shape, np.bool)
    sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = sum(y_test[i])
        for j in range(num):
            y_pred_new[i][sort_index[i][j]] = True
    return y_pred_new



from sklearn.cluster import KMeans
from sklearn import metrics

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

# def node_clustering(emb, one_hots):
#     label = [np.argmax(one_hot) for one_hot in one_hots]
#     ClusterNUm = np.unique(label)
#     clf = KMeans(n_clusters=len(ClusterNUm),init="k-means++")
#     kmeans = clf.fit(emb)
#
#     cluster_groups = kmeans.labels_
#     acc =acc_val(np.array(label),np.array(cluster_groups))
#     nmi = metrics.normalized_mutual_info_score(label,cluster_groups)
#     return acc,nmi

def node_clustering(emb, one_hots):
    label = [np.argmax(one_hot) for one_hot in one_hots]
    ClusterNUm = np.unique(label)

    model = WKMeans(n_clusters=len(ClusterNUm))

    cluster_groups = model.fit_predict(emb)


    acc =acc_val(np.array(label),np.array(cluster_groups))
    nmi = metrics.normalized_mutual_info_score(label,cluster_groups)
    return acc,nmi


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')

        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')

        return False





def cal_distance(x, y):
    return torch.sum((x-y)**2)**0.5








