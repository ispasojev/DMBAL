import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from src.helper.diversity_strategy_helper import compute_boundary_points, compute_boundary_points_overflow_canceled, \
    compute_boundary_prop_clustersize
import pandas as pd

from .strategy import Strategy


class BoundaryPoints(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, optimizer, args, diversity_strategy, k=1000, beta=1000, n_clusters=10,
                 uncertainty='margin', weighted=False):
        super(BoundaryPoints, self).__init__(X, Y, idxs_lb, net, handler, optimizer, args)
        self.beta = beta
        self.k = k
        self.n_clusters = n_clusters
        self.diversity_strategy = diversity_strategy
        self.uncertainty = uncertainty
        self.weighted = weighted

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]  # original unlabeled idx
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])  # softmax, idx verloren
        probs_sorted, labels = probs.sort(descending=True)  # sort class probs; idxs von diesen probs = labels

        if self.uncertainty == 'margin':
            margins = probs_sorted[:, 0] - probs_sorted[:, 1]  # margins
            margin_top_values, indices = torch.topk(margins, self.k * self.beta, largest=False)  # indices von top margins

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

        if self.diversity_strategy == "boundary_points":
            smallest_margin_indices = compute_boundary_points(distances, self.k) #n statt self.k überall wo self.k
        if (self.diversity_strategy == "bp_smo_canceled") | (self.diversity_strategy == "bp_lmo_canceled"):
            smallest_margin_indices = compute_boundary_points_overflow_canceled(distances, self.diversity_strategy,
                                                                                self.k, self.n_clusters)
        if self.diversity_strategy == "bp_prop_clustersize":
            smallest_margin_indices = compute_boundary_prop_clustersize(distances, self.k, self.n_clusters)

        original_indices_for_subset = idxs_unlabeled[indices[smallest_margin_indices]]

        #save which samples selected
        sampling_df = pd.DataFrame([list(original_indices_for_subset)], index=["img_id"]).T
        budget_data = {'idx': idxs_unlabeled[indices], 'centroid': budget_centroids}
        budget_df = pd.DataFrame(budget_data)
        self.save_stats(sampling_df)
        self.save_budget(budget_df)

        return original_indices_for_subset

def getCentroids(distances):
    centroids = []
    iter = 0
    for row in distances[:, 0]:
        centroid = np.argmin(distances[iter, :])
        centroids. append(centroid)
        iter += 1
    return centroids