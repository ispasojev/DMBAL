import numpy as np
import torch

from .strategy import Strategy


class RandomUncertainty(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, optimizer, args, k=1000, beta=10, uncertainty='margin'):
        super(RandomUncertainty, self).__init__(X, Y, idxs_lb, net, handler, optimizer, args)
        self.beta = beta
        self.k = k
        self.uncertainty = uncertainty

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, labels = probs.sort(descending=True)

        if self.uncertainty == 'margin':
            margins = probs_sorted[:, 0] - probs_sorted[:, 1]
            margin_top_values, indices = torch.topk(margins, self.k * self.beta, largest=False)

        if self.uncertainty == 'least_confident':
            least_conf_values, indices = torch.topk(probs_sorted[:, 0], self.k * self.beta, largest=False)

        if self.uncertainty == 'entropy':
            log_probs = torch.log(probs)
            entropies = (probs*log_probs).sum(1)
            entropy_values, indices = torch.topk(entropies, self.k * self.beta, largest=False)

        indices_numpy = indices.numpy()
        idxs_randomSelected = np.random.choice(indices_numpy, n, replace=False)
        chosen = idxs_unlabeled[idxs_randomSelected]
        return chosen