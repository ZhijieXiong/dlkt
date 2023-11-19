import torch
import numpy as np
from sklearn.cluster import KMeans


def cal_cosine_sim(A, B):
    # A是query，B是所有向量
    return A @ B.T / (np.linalg.norm(A, axis=1).reshape((-1, 1)) * np.linalg.norm(B.T, axis=0).reshape((1, -1)) + 1e-8)


class Cluster:
    def __init__(self, params):
        self.params = params
        self.num_cluster = params["other"]["cluster_cl"]["num_cluster"]

        self.clus = KMeans(n_clusters=self.num_cluster, n_init=5, max_iter=20)
        self.clus_center = None

    def train(self, X):
        self.clus.fit(X)

    def query(self, x_batch):
        cos_sim = cal_cosine_sim(x_batch, self.clus.cluster_centers_)
        seq2intent_id = np.argsort(cos_sim)[:, -1]
        seq2intent = self.clus.cluster_centers_[seq2intent_id]
        return (torch.LongTensor(seq2intent_id).to(self.params["device"]),
                torch.FloatTensor(seq2intent).to(self.params["device"]))
