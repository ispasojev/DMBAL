import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment

from .strategy import Strategy


class Random(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, optimizer, args):
        super(Random, self).__init__(X, Y, idxs_lb, net, handler, optimizer, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]  # original unlabeled idx
        shuffle_pool = idxs_unlabeled
        np.random.shuffle(shuffle_pool)
        chosen = shuffle_pool[:n]
        return chosen