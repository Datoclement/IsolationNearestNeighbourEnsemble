"""
    Isolation Nearest Neighbour Ensemble from 
    http://www.vincentlemaire-labs.fr/ICDM2014/slides/Ting.pdf
"""

from sklearn.neighbors import KDTree
import numpy as np

class INNE:
    """
        Isolation Nearest Neighbour Ensemble from 
        http://www.vincentlemaire-labs.fr/ICDM2014/slides/Ting.pdf
    """

    def __init__(self, sample_size=128, num_ensemble=100, contamination=.1):
        """
            sample_ratio: int. subset size
            num_ensemble: int. ensemble size
        """

        self.r = sample_size
        self.E = num_ensemble
        self.contamination = contamination
        self.thresh = None
        self.estimators = list()
        self.benchmark = dict()

    def fit(self, X, y=None):
        """
            X: np.ndarray. of shape (number_of_samples(n), number_of_features(d))
            y: ignored. for API consistency
        """
        self.X = X
        n, d = X.shape
        m = self.r
        assert m <= n

        for _ in range(self.E):
            idcs = np.random.permutation(len(X))[:m]
            xs = X[idcs]
            kdtr = KDTree(xs)
            dists, nnids = kdtr.query(xs, k=2)
            dists = dists[:, 1]
            nnids = nnids[:, 1]
            nndists = dists[nnids]
            epsdists = dists + (dists == 0.).astype(float)
            scores = 1. - nndists / epsdists
            self.estimators.append([X[None, idcs], dists[None], scores[None]])

        return self

    def predict(self, X):
        """
            X: np.ndarray. of shape (n', d)

            -1: is anomaly
            +1: is normal
        """
        if self.thresh is None:
            self.thresh = np.percentile(self.score_samples(self.X), 100 * (1.-self.contamination))
        is_anomaly = (self.score_samples(X) > self.thresh).astype(int)
        return 1 - 2 * is_anomaly

    def score_samples(self, X):
        """
            X: np.ndarray. of shape (n', d)

            The higher the score is, the more normal the sample is.
        """

        # compute pairwise distance with (x-y)**2 = x**2 + y**2 - 2*x*y
        xx = np.sum(X**2, -1)[:, None]

        score_board = list()
        for _, (samples, radius, scores) in enumerate(self.estimators):

            yy = np.sum(samples**2, -1)
            xy = np.dot(X, samples[0].T)
            dists = xx + yy - 2 * xy
            radius = radius**2
            in_balls = (dists <= radius).astype(float)
            tmp_radius = radius * in_balls + (1.-in_balls) * radius.max(-1, keepdims=True)
            true_scores = scores * in_balls + (1.-in_balls)
            score_board.append(
                true_scores[list(range(true_scores.shape[0])), tmp_radius.argmin(-1)])

        scores = 1. - np.array(score_board).mean(0)
        return scores
