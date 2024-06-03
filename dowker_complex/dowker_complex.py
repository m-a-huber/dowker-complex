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
        max_dimension=2
    ):
        self.metric = metric
        self.max_dimension = max_dimension

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
        indicate_outliers=True,
        indicate_labels=False,
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
            indicate_outliers=indicate_outliers,
            indicate_labels=indicate_labels,
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
        distances = np.concatenate([[0], np.unique(self._dm_)])
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
        for dist_ix, dist in enumerate(np.unique(self._dm_)):
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

    def _get_complex(self):
        batches = self._get_batches()
        st = SimplexTree()
        for vertex_array, filtrations in batches:
            st.insert_batch(
                vertex_array=vertex_array.astype(int),
                filtrations=filtrations
            )
        return st

    def _get_batches(self):
        self._dm_ = pairwise_distances(self.W_, self.V_)
        distances = np.unique(self._dm_)
        simplices = np.array([
                self._dm_ <= distance
                for distance in distances
        ])
        simplices_with_filtrations = np.concatenate([
            np.vstack(simplices),
            np.repeat(distances, len(self.W_)).reshape(-1, 1)
        ], axis=1)

        def _batches(dim):
            ixs = np.argwhere(
                np.sum(
                    simplices_with_filtrations[:, :-1].astype(int),
                    axis=1
                ) == dim+1
            ).reshape(-1,)
            vertex_array, filtrations = (
                simplices_with_filtrations[ixs][:, :-1],
                simplices_with_filtrations[ixs][:, -1]
            )
            vertex_array = np.nonzero(vertex_array)[1].reshape(-1, dim+1).T
            return vertex_array, filtrations
        yield from (_batches(dim) for dim in range(self.max_dimension+1))

    # def _check_filtrations(self):
    #     return (
    #         self.filtrations_.diagonal() == self.filtrations_.min(axis=1)
    #     ).all()

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
