import warnings
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
import numpy as np
import plotly.graph_objects as gobj
from shapely.geometry import MultiPoint
from gudhi import SimplexTree
from datasets_custom.plotting import plot_point_cloud, plot_persistences
import pandas as pd


class DowkerComplex(BaseEstimator):
    """This class implements the Dowker persistent homology introduced in [1].
    This, in turn, is a generalization of the Dowker complex introduced in [2]
    to the setting of persistent homology.

    Parameters:
        metric (str, optional): The metric used to compute distance between
            data points. Must be one of the metrics listed in
            ``sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
            Defaults to "euclidean".
        max_dimension(int, optional): The maximum dimension of simplices used
            when creating the Dowker simplicial complex. Defaults to 2.
        max_filtration (float, optional): The maximum filtration value of
            simplices used when creating the Dowker simplicial complex.
            Defaults to `np.inf`.

    Attributes:
        vertices_ (numpy.ndarray of shape (n_vertices, dim)): NumPy-array
            containing the vertices.
        witnesses_ (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
            containing the witnesses.
        filtrations_ (numpy.ndarray of shape (n_unique_distances,)):
            NumPy-array containing the unique distances between vertices and
            witnesses, sorted in ascending order.
        simplices_ (dict of int: dict of str: numpy.ndarray): Dictionary whose
            keys are the integers 0, ..., `max_dimension`, and whose values
            are dictionaries containing the arguments to
            `gudhi.SimplexTree.insert_batch`. That is, each of these
            dictionaries has `"vertex_array"` and `"filtrations"` as keys, and
            NumPy-arrays of shape (dim + 1, n_simplices) and (n_simplices,),
            respectively, as its values.
        complex_ (:class:`~gudhi.SimplexTree`): The Dowker simplicial complex
            constructed from the vertices and witnesses, given as an instance
            of :class:`~gudhi.SimplexTree`.
        persistence_ (list[numpy.ndarray]): The persistent homology computed
            from the Dowker simplicial complex. The format of this data is a
            list of NumPy-arrays of shape (n_generators, 2), where the i-th
            entry of the list is an array containing the birth and death times
            of the homological generators in dimension i-1. In particular, the
            list starts with 0-dimensional homology and contains information
            from consecutive homological dimensions.

    References:
        [1]: Samir Chowdhury, & Facundo Mémoli (2018). A functorial Dowker
            theorem and persistent homology of asymmetric networks. J. Appl.
            Comput. Topol., 2(1-2), 115–175.
        [2]: C. H. Dowker (1952). Homology Groups of Relations. Annals of
            Mathematics, 56(1), 84–95.

    Examples:
    """
    def __init__(
        self,
        metric="euclidean",
        max_dimension=2,
        max_filtration=np.inf
    ):
        self.metric = metric
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration

    def fit(
        self,
        vertices,
        witnesses,
        compute_persistence=True,
        **persistence_kwargs
    ):
        """Method that fits an DowkerComplex instance to a pair of point
        clouds consisting of vertices and witnesses.

        Args:
            vertices (numpy.ndarray of shape (n_vertices, dim)): NumPy-array
                containing the vertices.
            witnesses (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
                containing the witnesses.
            compute_persistence (bool, optional): Whether or not persistent
                homology should be computed, as opposed to computing the
                Dowker simplicial complex only. Defaults to True.

        Returns:
            :class:`dowker_complex.DowkerComplex`: Fitted instance of
                DowkerComplex.
        """
        self.vertices_ = vertices
        self.witnesses_ = witnesses
        self._labels_vertices_ = np.zeros(len(self.vertices_))
        self._labels_witnesses_ = -np.ones(len(self.witnesses_))
        self._points_ = np.concatenate([self.vertices_, self.witnesses_])
        self._labels_ = np.concatenate([
            self._labels_vertices_,
            self._labels_witnesses_
        ])
        self.complex_ = self._get_complex()
        if self.max_filtration < np.inf:
            self.complex_.prune_above_filtration(self.max_filtration)
        if compute_persistence:
            persistence = self.complex_.persistence(**persistence_kwargs)
            self.persistence_ = self._format_persistence(persistence)
        return self

    def _get_complex(self):
        self._dm_ = pairwise_distances(
            self.witnesses_,
            self.vertices_,
            metric=self.metric
        )
        self.filtrations_ = np.unique(self._dm_)
        self._vertex_ixs_ = np.array([
                self._dm_ <= filtration
                for filtration in self.filtrations_
        ]).astype(int)
        self._vertex_ixs_to_filtration_ = np.concatenate(
            [
                np.concatenate(self._vertex_ixs_),
                np.repeat(
                    self.filtrations_,
                    len(self.witnesses_)
                ).reshape(-1, 1)
            ],
            axis=1
        )
        self._vertex_ixs_to_filtration_grouped_ = self._group_by_last_col(
            self._vertex_ixs_to_filtration_
        )
        self._splits_ = self._get_splits(
            self._vertex_ixs_to_filtration_grouped_
        )
        simplices_list = [
            self._group_by_last_col(
                self._get_simplices(dim=dim)
            )
            for dim in range(self.max_dimension+1)
        ]
        self.simplices_ = {
            dim: {
                "vertex_array": np.transpose(simplices[:, :-1]).astype(int),
                "filtrations": simplices[:, -1]
            }
            for dim, simplices in enumerate(simplices_list)
        }
        simplex_tree_ = SimplexTree()
        for dim in range(self.max_dimension+1):
            simplex_tree_.insert_batch(
                **self.simplices_[dim]
            )
        return simplex_tree_

    @staticmethod
    def _group_by_last_col(a):
        def is_sorted(a): return np.all(a[:-1] <= a[1:])
        if not is_sorted(a[:, -1]):
            a = a[np.argsort(a[:, -1])]
        df = pd.DataFrame(a)
        cols = df.columns.to_list()
        df = df.groupby(
            cols[:-1],
            as_index=False
        )[cols[-1]].agg("min")
        return df.to_numpy()

    @staticmethod
    def _get_splits(arr):
        return [
            arr[np.sum(arr[:, :-1], axis=1) == k]
            for k in range(1, arr.shape[1])
        ]

    def _get_simplices(self, dim=1):
        return np.concatenate(
            [
                np.concatenate(
                    [
                        self._get_ixs_batch(split[:, :-1], dim=dim),
                        np.repeat(
                            split[:, -1],
                            self._binom(i+1, dim+1)
                        ).reshape(-1, 1)
                    ],
                    axis=1
                )
                for i, split in enumerate(self._splits_)
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
    def _binom(n, k):
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
        """Method plotting the Dowker persistence. Underlying instance must be
        fitted and have the attribute `persistence_`.

        Args:
            plotting_kwargs (optional): Arguments passed to the function
                `datasets_custom.persistence_plotting.plot_persistences`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                persistence diagram.
        """
        check_is_fitted(self, attributes="persistence_")
        fig = plot_persistences(
            [self.persistence_],
            **plotting_kwargs
        )
        return fig

    def plot_points(
        self,
        indicate_witnesses=True,
        use_colors=True,
        **plotting_kwargs
    ):
        """Method plotting the vertices and witnesses underlying a fitted
        instance of DowkerComplex. Works for point clouds up to dimension
        three only.

        Args:
            indicate_witnesses (bool, optional): Whether or not to use a
                distinguished marker to indicate the witness points.
                Defaults to True.
            use_colors (bool, optional): Whether or not to color the vertices
                and witnesses in different colors. Defaults to True.
            plotting_kwargs (optional): Arguments passed to the function
                `datasets_custom.utils.plotting.plot_point_cloud`, such as
                `marker_size` and `colorscale`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                vertex and witness point clouds.
        """
        check_is_fitted(self, attributes=["vertices_", "witnesses_"])
        if self._points_.shape[1] not in {1, 2, 3}:
            raise Exception(
                "Plotting is supported only for data "
                "sets of dimension at most 3."
            )
        return plot_point_cloud(
            self._points_,
            labels=self._labels_,
            indicate_outliers=indicate_witnesses,
            indicate_labels=use_colors,
            colorscale="wong",
            **plotting_kwargs
        )

    def plot_skeleton(
        self,
        k=2,
        threshold=np.inf,
        indicate_witnesses=True,
        use_colors=True,
        **plotting_kwargs
    ):
        """Method plotting the k-skeleton of the Dowker complex underlying a
        fitted instance of DowkerComplex. Works for values of k and point
        clouds of dimension up to and including 2.

        Args:
            k (int, optional): Dimension of the skeleton to be plotted.
                Defaults to 2.
            threshold (_type_, optional): The maximum filtration level of
                simplices to be plotted. Defaults to np.inf.
            indicate_witnesses (bool, optional): Whether or not to use a
                distinguished marker to indicate the witness points.
                Defaults to True.
            use_colors (bool, optional): Whether or not to color the vertices
                and witnesses in different colors. Defaults to True.
            plotting_kwargs (optional): Arguments passed to the function
                `datasets_custom.utils.plotting.plot_point_cloud`, such as
                `line_width` and `colorscale`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                k-skeleton.
        """
        if self._points_.shape[1] not in {1, 2}:
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
        vertex_ixs = [
            simplex[0]
            for simplex, filtration in complex.get_skeleton(dimension=1)
            if len(simplex) == 1
        ]
        points = np.concatenate([
            self.vertices_[vertex_ixs].reshape(-1, 2),
            self.witnesses_
        ])
        labels = np.concatenate([
            self._labels_vertices_[vertex_ixs],
            self._labels_witnesses_
        ])
        if k >= 1:
            lines = self._points_[[
                simplex
                for simplex, filtration in complex.get_skeleton(dimension=1)
                if len(simplex) == 2
            ]].reshape(-1, 2, 2)
        fig = plot_point_cloud(
            points,
            labels=labels,
            indicate_outliers=indicate_witnesses,
            indicate_labels=use_colors,
            colorscale="wong",
            lines=lines if k >= 1 else None,
            **plotting_kwargs
        )
        if k == 2:
            two_simplices_with_filtration = [
                (spx, filtration)
                for spx, filtration in complex.get_skeleton(dimension=2)
                if len(spx) >= 3
            ]
            two_simplices = np.array([
                self.vertices_[two_simplex]
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
            self._points_
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
        indicate_witnesses=True,
        use_colors=True,
        **plotting_kwargs
    ):
        """Method plotting an interactive version of the k-skeleton of the
        Dowker complex underlying a fitted instance of DowkerComplex. Works
        for values of k and point clouds of dimension up to and including 2.

        Args:
            k (int, optional): Dimension of the skeleton to be plotted.
                Defaults to 2.
            indicate_witnesses (bool, optional): Whether or not to use a
                distinguished marker to indicate the witness points.
                Defaults to True.
            use_colors (bool, optional): Whether or not to color the vertices
                and witnesses in different colors. Defaults to True.
            plotting_kwargs (optional): Arguments passed to the function
                `datasets_custom.utils.plotting.plot_point_cloud`, such as
                `line_width` and `colorscale`.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: An interactive plot of
                the k-skeleton.
        """
        if self._points_.shape[1] not in {1, 2}:
            raise Exception(
                "Plotting of the skeleton is supported only "
                "for data sets of dimension at most 2."
            )
        if k not in {0, 1, 2}:
            raise Exception(
                "The value of `k` must be either 0, 1 or 2."
            )
        fig_combined = self.plot_skeleton(
            threshold=0,
            k=0,
            indicate_witnesses=indicate_witnesses,
            use_colors=use_colors,
            **plotting_kwargs
        )
        distances = np.concatenate([
            [0],
            np.unique([
                filtration
                for simplex, filtration in self.complex_.get_filtration()
            ])
        ])
        print(f"{len(distances) = }")
        datum_ixs = defaultdict(list)
        datum_ix = len(fig_combined.data)
        for dist_ix, dist in enumerate(distances):
            fig = self.plot_skeleton(
                threshold=dist,
                k=k,
                indicate_witnesses=indicate_witnesses,
                use_colors=use_colors,
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
