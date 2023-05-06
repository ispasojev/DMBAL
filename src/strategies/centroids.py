import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment

from .strategy import Strategy


class Centroids(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, optimizer, args, k=10, beta=10, n_clusters=1000,
                 uncertainty='margin', weighted=False):
        super(Centroids, self).__init__(X, Y, idxs_lb, net, handler, optimizer, args)
        self.beta = beta
        self.k = k
        self.n_clusters = n_clusters
        self.uncertainty = uncertainty
        self.weighted = weighted

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]  # original unlabeled idx
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])  # probabilities (with softmax)
        probs_sorted, labels = probs.sort(descending=True)  # sort class probs

        if self.uncertainty == 'margin':
            margins = probs_sorted[:, 0] - probs_sorted[:, 1]  # margins
            margin_top_values, indices = torch.topk(margins, self.k * self.beta, largest=False)  # indices von top margins
            print(len(indices))
            print("len(indices)")

        if self.uncertainty == 'least_confident':
            least_conf_values, indices = torch.topk(probs_sorted[:, 0], self.k * self.beta, largest=False)

        if self.uncertainty == 'entropy':
            log_probs = torch.log(probs)
            entropies = (probs*log_probs).sum(1)
            entropy_values, indices = torch.topk(entropies, self.k * self.beta, largest=False)

        embedding = self.get_embedding(self.X[idxs_unlabeled[indices]], self.Y[idxs_unlabeled[indices]])
        embedding = embedding.reshape(self.k * self.beta, -1).numpy()
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=10)

        if self.weighted:
            if self.uncertainty == 'margin':
                distances = kmeans.fit_transform(embedding, y=None, sample_weight=margin_top_values)
            if self.uncertainty == 'least_confident':
                distances = kmeans.fit_transform(embedding, y=None, sample_weight=least_conf_values)
            if self.uncertainty == 'entropy':
                distances = kmeans.fit_transform(embedding, y=None, sample_weight=entropy_values)
        else:
            distances = kmeans.fit_transform(embedding)
        budget_centroids = getCentroids(distances)

        nn_to_centroids_indices = linear_sum_assignment(distances)[0]  # takes the closest to each center
        chosen = idxs_unlabeled[indices[nn_to_centroids_indices]]

        # save labeled samples
        sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
        budget_data = {'idx': idxs_unlabeled[indices], 'centroid': budget_centroids}
        budget_df = pd.DataFrame(budget_data)
        self.save_stats(sampling_df)
        self.save_budget(budget_df)
        return chosen

def getCentroids(distances):
    centroids = []
    iter = 0
    for row in distances[:, 0]:
        centroid = np.argmin(distances[iter, :])
        centroids. append(centroid)
        iter += 1
    return centroids
