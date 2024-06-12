import warnings
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
import numpy as np
import plotly.graph_objects as gobj
from shapely.geometry import MultiPoint
from gudhi import SimplexTree
from datasets_custom.utils.plotting import plot_point_cloud
from datasets_custom.persistence_plotting import plot_persistences


class DowkerComplex(BaseEstimator):

    def __init__(
        self,
        metric="euclidean",
        max_dimension=2,
        max_filtration=np.inf
    ):
        self.metric = metric
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration

    # V are potential vertices, W are reference points
    def fit(
        self,
        V,
        W,
        max_filtration=np.inf,
        compute_persistence=True,
        **persistence_kwargs
    ):
        self.V_ = V
        self.W_ = W
        self.points_ = np.concatenate([self.V_, self.W_])
        self.v_labels_ = np.zeros(len(self.V_))
        self.w_labels_ = -np.ones(len(self.W_))
        self.labels_ = np.concatenate([
            self.v_labels_,
            self.w_labels_
        ])
        self.complex_ = self._get_complex()
        if compute_persistence:
            persistence = self.complex_.persistence(**persistence_kwargs)
            self.persistence_ = self._format_persistence(persistence)
        return self

    def _get_complex(self):
        self._dm_ = pairwise_distances(self.W_, self.V_)
        self.filtrations_ = np.unique(self._dm_)
        self.vertex_ixs = np.array([
                self._dm_ <= filtration
                for filtration in self.filtrations_
        ]).astype(int)
        self.vertex_ixs_to_filtration = np.concatenate(
            [
                np.concatenate(self.vertex_ixs),
                np.repeat(self.filtrations_, len(self.W_)).reshape(-1, 1)
            ],
            axis=1
        )
        self.vertex_ixs_to_filtration_grouped = self._group_by_last_col(
            self.vertex_ixs_to_filtration
        )
        self._splits = self._get_splits(self.vertex_ixs_to_filtration_grouped)
        simplices_list = [
            self._group_by_last_col(
                self._get_simplices(dim=dim)
            )
            for dim in range(self.max_dimension+1)
        ]
        self.simplices = {
            dim: {
                "vertex_array": np.transpose(simplices[:, :-1]).astype(int),
                "filtrations": simplices[:, -1]
            }
            for dim, simplices in enumerate(simplices_list)
        }
        self.simplex_tree = SimplexTree()
        for dim in range(self.max_dimension+1):
            self.simplex_tree.insert_batch(
                **self.simplices[dim]
            )
        return self.simplex_tree

    @staticmethod
    def _group_by_last_col(a):
        def is_sorted(a): return np.all(a[:-1] <= a[1:])
        if not is_sorted(a[:, -1]):
            a = a[np.argsort(a[:, -1])]
        vals = np.unique(a[:, :-1], axis=0)
        ixs = np.array([
            np.argmax(
                np.all(a[:, :-1].astype(int) == val, axis=1),
                axis=0
            )
            for val in vals
        ])
        return a[ixs]

    @staticmethod
    def _get_splits(arr):
        return [
            arr[np.sum(arr[:, :-1], axis=1) == k]
            for k in range(2, arr.shape[1])
        ]

    def _get_simplices(self, dim=1):
        return np.concatenate(
            [
                np.concatenate(
                    [
                        self._get_ixs_batch(split[:, :-1], dim=dim),
                        np.repeat(
                            split[:, -1],
                            self.binom(i+2, dim+1)
                        ).reshape(-1, 1)
                    ],
                    axis=1
                )
                for i, split in enumerate(self._splits)
            ],
            axis=0
        )

    @staticmethod
    def _get_ixs_batch(batch, dim=1):
        if batch.shape[0] == 0:
            return np.array([]).reshape(0, dim+1)
        if dim == 1:
            batch_one = np.nonzero(batch)[1].reshape(batch.shape[0], -1)
            ixs = np.transpose(np.triu_indices(batch_one.shape[1], 1))
            return np.concatenate(batch_one[:, ixs], axis=0)
        else:
            def _triu_cust(n, d):
                if d == 1:
                    return np.arange(n).reshape(-1, 1)
                aux = np.transpose(np.triu(np.ones((n,) * d)).nonzero())
                return aux[np.all(aux[:, :-1] < aux[:, 1:], axis=1)]
            batch_one = np.nonzero(batch)[1].reshape(batch.shape[0], -1)
            ixs = _triu_cust(batch_one.shape[1], dim+1)
            return np.concatenate(batch_one[:, ixs], axis=0)

    @staticmethod
    def binom(n, k):
        if k == 1:
            return n
        if k == 2:
            return (n*(n-1))//2
        elif k == 3:
            return (n*(n-1)*(n-2))//6
        else:
            import scipy
            return int(scipy.special.binom(n, k))

    @staticmethod
    def _format_persistence(persistence):
        if len(persistence) == 0:
            max_hom_dim = 0
        else:
            max_hom_dim = max([dim for dim, gen in persistence])
        persistence_formatted = [
            np.array([
                gen
                for dim, gen in persistence if dim == i
            ]).reshape(-1, 2)
            for i in range(max_hom_dim+1)
        ]
        persistence_sorted = [
            hom[np.argsort(np.diff(hom, axis=1).reshape(-1,))]
            for hom in persistence_formatted
        ]
        return persistence_sorted

    def plot_persistence(self, **plotting_kwargs):
        check_is_fitted(self, attributes="persistence_")
        fig = plot_persistences(
            [self.persistence_],
            **plotting_kwargs
        )
        return fig

    def plot_points(
        self,
        indicate_outliers=True,
        indicate_labels=False,
        **plotting_kwargs
    ):
        if self.points_.shape[1] not in {1, 2, 3}:
            raise Exception(
                "Plotting is supported only for data "
                "sets of dimension at most 3."
            )
        return plot_point_cloud(
            self.points_,
            labels=self.labels_,
            indicate_outliers=indicate_outliers,
            indicate_labels=indicate_labels,
            **plotting_kwargs
        )

    def plot_skeleton(
        self,
        k=2,
        threshold=np.inf,
        line_width=1,
        colorscale="jet",
        **plotting_kwargs
    ):
        if self.points_.shape[1] not in {1, 2}:
            raise Exception(
                "Plotting of the skeleton is supported only "
                "for data sets of dimension at most 2."
            )
        if k not in {0, 1, 2}:
            raise Exception(
                "The value of `k` must be either 0, 1 or 2."
            )
        check_is_fitted(self, attributes="complex_")
        complex = self.complex_.copy()
        complex.prune_above_filtration(threshold)
        point_ixs = [
            t[0][0]
            for t in complex.get_skeleton(dimension=1)
            if len(t[0]) == 1
        ]
        points = np.concatenate([
            self.V_[point_ixs].reshape(-1, 2),
            self.W_
        ])
        labels = np.concatenate([
            self.v_labels_[point_ixs],
            self.w_labels_
        ])
        if k >= 1:
            lines = self.points_[[
                t[0]
                for t in complex.get_skeleton(dimension=1)
                if len(t[0]) == 2
            ]].reshape(-1, 2, 2)
        fig = plot_point_cloud(
            points,
            labels=labels,
            lines=lines if k >= 1 else None,
            line_width=line_width,
            colorscale=colorscale,
            **plotting_kwargs
        )
        if k == 2:
            two_simplices_with_filtration = [
                (spx, filtration)
                for spx, filtration in complex.get_skeleton(dimension=2)
                if len(spx) >= 3
            ]
            two_simplices = np.array([
                self.V_[two_simplex]
                for two_simplex, filtration in two_simplices_with_filtration
            ])
            for two_simplex in two_simplices:
                x = two_simplex.T[0]
                y = two_simplex.T[1]
                convex_hull = np.array(
                    MultiPoint(
                        [xy for xy in zip(x, y)]
                    ).convex_hull.exterior.coords
                )
                polygon = gobj.Scatter(
                    x=convex_hull[:, 0],
                    y=convex_hull[:, 1],
                    showlegend=False,
                    mode="lines",
                    fill="toself",
                    opacity=0.3,
                    fillcolor="grey",
                    line_color="grey"
                )
                fig.add_trace(polygon)
        fig_ref = plot_point_cloud(
            self.points_
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            xrange = fig_ref.full_figure_for_development().layout.xaxis.range
            yrange = fig_ref.full_figure_for_development().layout.yaxis.range
        fig.update_layout(
            xaxis_range=xrange,
            yaxis_range=yrange
        )
        return fig

    def plot_interactive_skeleton(
        self,
        k=2,
        line_width=1,
        indicate_outliers=True,
        indicate_labels=False,
        **plotting_kwargs
    ):
        if k not in {0, 1, 2}:
            raise Exception(
                "The value of `k` must be either 0, 1 or 2."
            )
        fig_combined = self.plot_skeleton(
            threshold=0,
            k=0,
            line_width=line_width,
            indicate_outliers=indicate_outliers,
            indicate_labels=indicate_labels,
            **plotting_kwargs
        )
        distances = np.unique([
            filtration
            for splx, filtration in self.simplex_tree.get_filtration()
        ])
        datum_ixs = defaultdict(list)
        datum_ix = len(fig_combined.data)
        for dist_ix, dist in enumerate(distances):
            fig = self.plot_skeleton(
                threshold=dist,
                k=k,
                line_width=line_width,
                indicate_outliers=indicate_outliers,
                indicate_labels=indicate_labels,
                **plotting_kwargs
            )
            for datum in fig.data:
                fig_combined.add_trace(datum)
                datum_ixs[dist_ix].append(datum_ix)
                datum_ix += 1
        steps = []
        for dist_ix, dist in enumerate(distances):
            step = dict(
                method="update",
                args=[
                    {"visible": [False] * len(fig_combined.data)},
                ],
                label=str(np.round(distances[dist_ix], 6))
            )
            for ix in datum_ixs[dist_ix]:
                step["args"][0]["visible"][ix] = True
            steps.append(step)
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Threshold: "},
            pad={"t": 50},
            steps=steps
        )]
        fig_combined.update_layout(
            sliders=sliders
        )
        return fig_combined
