from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
import numpy as np
from gudhi import SimplexTree


class DowkerComplex(BaseEstimator):

    def __init__(self, metric="euclidean"):
        self.metric = metric

    # V are potential vertices, W are reference points
    def fit(
        self,
        V,
        W,
        max_dimension=2,
        max_filtration=np.inf,
        **persistence_kwargs
    ):
        self.V_ = V
        self.W_ = W
        self.filtrations_ = self._get_filtrations()
        assert self._check_filtrations()
        self.complex_ = SimplexTree.create_from_array(
            self.filtrations_,
            max_filtration
        )
        self.complex_.expansion(max_dimension=max_dimension)
        persistence = self.complex_.persistence(**persistence_kwargs)
        self.persistence_ = self._format_persistence(persistence)
        return self

    def _get_filtrations(self):
        n_V = len(self.V_)
        X = np.concatenate([self.V_, self.W_])
        dm = pairwise_distances(X)
        vertex_filtrations = np.min(
            dm[:n_V, n_V:], axis=1
        )
        edge_filtrations = dm[:n_V, :n_V]
        filtrations = edge_filtrations
        np.fill_diagonal(filtrations, vertex_filtrations)
        diag = filtrations.diagonal()
        diag_ver = np.tile(diag, (len(diag), 1))
        diag_hor = diag_ver.T
        filtrations = np.fmax(filtrations, np.fmax(diag_ver, diag_hor))
        return filtrations

    def _check_filtrations(self):
        return (
            self.filtrations_.diagonal() == self.filtrations_.min(axis=1)
        ).all()

    @staticmethod
    def _format_persistence(persistence):
        persistence_formatted = [
            np.array([
                gen
                for dim, gen in persistence if dim == i
            ])
            for i in range(max([dim for dim, gen in persistence])+1)
        ]
        persistence_sorted = [
            hom[np.argsort(np.diff(hom, axis=1).reshape(-1,))]
            for hom in persistence_formatted
        ]
        return persistence_sorted
