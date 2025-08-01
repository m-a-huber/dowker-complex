import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as gobj  # type: ignore
from gudhi import SimplexTree  # type: ignore
from numba import jit, prange  # type: ignore
from shapely.geometry import MultiPoint  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from typing_extensions import Self

from .plotting.persistence_plotting import plot_persistences  # type: ignore
from .plotting.point_cloud_plotting import plot_point_cloud  # type: ignore


class DowkerComplex(TransformerMixin, BaseEstimator):
    """Class implementing the Dowker persistent homology associated to a
    point cloud whose elements are separated into two classes. This is
    introduced in [1] and is a generalization of the Dowker complex introduced
    in [2] to the setting of persistent homology. The data points on which the
    underlying simplicial complex is constructed are referred to as "vertices",
    while the other ones are referred to as "witnesses".

    Parameters:
        max_dimension (int, optional): The maximum homology dimension computed.
            Will compute all dimensions lower than or equal to this value.
            Currently, only values less than or equal to `1` are supported.
            Defaults to `1`.
        max_filtration (float, optional): The Maximum value of the Dowker
            filtration parameter. If `np.inf`, the entire filtration is
            computed. Defaults to `np.inf`.
        coeff (int, optional): The field coefficient used in the computation of
            homoology. Defaults to `2`.
        metric (str, optional): The metric used to compute distance between
            data points. Must be one of the metrics listed in
            ``sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
            Defaults to `"euclidean"`.
        metric_params (dict, optional): Additional parameters to be passed to
            the distance function. Defaults to `dict()`.
        swap (bool, optional): Whether or not to potentially swap the roles of
            vertices and witnesses to compute the less expensive variant of
            persistent homology. Defaults to `True`.
        verbose (bool, optional): Whether or not to display information such as
            computation progress. Defaults to `False`.

    Attributes:
        vertices_ (numpy.ndarray of shape (n_vertices, dim)): NumPy-array
            containing the vertices.
        witnesses_ (numpy.ndarray of shape (n_witnesses, dim)): NumPy-array
            containing the witnesses.
        simplices_ (dict of int: dict of str: numpy.ndarray): Dictionary whose
            keys are the integers 0, ..., `max_dimension + 1`, and whose values
            are dictionaries containing the arguments to
            `gudhi.SimplexTree.insert_batch`. That is, each of these
            dictionaries has `"vertex_array"` and `"filtrations"` as keys, and
            NumPy-arrays of shape (dim + 1, n_simplices) and (n_simplices,),
            respectively, as its values.
        complex_ (gudhi.SimplexTree): The Dowker simplicial complex constructed
            from the vertices and witnesses.
        persistence_ (list[numpy.ndarray]): The persistent homology computed
            from the Dowker simplicial complex. The format of this data is a
            list of NumPy-arrays of dtype float64 of shape `(n_generators, 2)`,
            where the i-th entry of the list is an array containing the birth
            and death times of the homological generators in dimension i-1. In
            particular, the list starts with 0-dimensional homology and
            contains information from consecutive homological dimensions.

    References:
        [1]: Samir Chowdhury, & Facundo Mémoli (2018). A functorial Dowker
            theorem and persistent homology of asymmetric networks. J. Appl.
            Comput. Topol., 2(1-2), 115-175.
        [2]: C. H. Dowker (1952). Homology Groups of Relations. Annals of
            Mathematics, 56(1), 84-95.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        max_filtration: float = np.inf,
        coeff: int = 2,
        metric: str = "euclidean",
        metric_params: dict = dict(),
        swap: bool = True,
        verbose: bool = False,
    ) -> None:
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration
        self.coeff = coeff
        self.metric = metric
        self.metric_params = metric_params
        self.swap = swap
        self.verbose = verbose

    def vprint(
        self,
        s: str,
    ) -> None:
        if self.verbose:
            print(s)
        return

    def fit(
        self,
        X: list[npt.NDArray],
        y: Optional[None] = None,
    ) -> Self:
        """Method that fits a `DowkerComplex`-instance to a pair of point
        clouds consisting of vertices and witnesses by constructing the
        associated Dowker complex, as an instance of `gudhi.SimplexTree`.

        Args:
            X (list[numpy.ndarray]): List containing the NumPy-arrays of
                vertices and witnesses, in this order.
            y (None, optional): Not used, present here for API consistency with
                scikit-learn.

        Returns:
            self (DowkerComplex): The fitted instance of `DowkerComplex`.
        """
        if self.max_dimension > 1:
            raise ValueError(
                f"The value for `max_dimension` is `{self.max_dimension}`, "
                "but only values less than or equal to `1` are supported."
            )
        vertices, witnesses = X
        if vertices.shape[1] != witnesses.shape[1]:
            raise ValueError(
                "The vertices and witnesses should be of the same "
                f"dimensionality; received dim(vertices)={vertices.shape[1]} "
                f"and dim(witnesses)={witnesses.shape[1]}."
            )
        if self.swap and len(vertices) > len(witnesses):
            vertices, witnesses = witnesses, vertices
            self.vprint("Swapped roles of vertices and witnesses.")
        self.vertices_ = vertices
        self.witnesses_ = witnesses
        self.vprint(
            "Complex has (n_vertices, n_witnesses) = "
            f"{(len(self.vertices_), len(self.witnesses_))}."
        )
        self._labels_vertices_ = np.zeros(len(self.vertices_))
        self._labels_witnesses_ = -np.ones(len(self.witnesses_))
        self._points_ = np.concatenate([self.vertices_, self.witnesses_])
        self._labels_ = np.concatenate(
            [self._labels_vertices_, self._labels_witnesses_]
        )
        if min(len(self.vertices_), len(self.witnesses_)) == 0:
            self.complex_ = SimplexTree()
        else:
            self.complex_ = self._get_complex()
        return self

    def transform(
        self,
        X: list[npt.NDArray],
        y: Optional[None] = None,
    ) -> list[npt.NDArray[np.float64]]:
        """Method that transforms a `DowkerComplex`-instance fitted to a pair
        of point clouds consisting of vertices and witnesses by computing the
        persistent homology of the associated Dowker complex.

        Args:
            X (list[numpy.ndarray]): List containing the NumPy-arrays of
                vertices and witnesses, in this order.
            y (None, optional): Not used, present here for API consistency with
                scikit-learn.

        Returns:
            list[numpy.ndarray]: The persistent homology computed from the
                Dowker simplicial complex. The format of this data is a list of
                NumPy-arrays of dtype float64 and of shape `(n_generators, 2)`,
                where the i-th entry of the list is an array containing the
                birth and death times of the homological generators in
                dimension i-1. In particular, the list starts with
                0-dimensional homology and contains information from
                consecutive homological dimensions.
        """
        check_is_fitted(self, attributes="complex_")
        self.vprint("Computing persistent homology...")
        persistence_dim_max = self.complex_.dimension() <= self.max_dimension,
        self.persistence_ = self._format_persistence(
            self.complex_.persistence(
                homology_coeff_field=self.coeff,
                min_persistence=0.0,
                persistence_dim_max=persistence_dim_max
            )
        )
        self.vprint("Done computing persistent homology.")
        return self.persistence_

    def _get_complex(
        self
    ):
        self._dm_ = pairwise_distances(
            self.vertices_, self.witnesses_, metric=self.metric
        )
        self.vprint("Getting simplices...")
        self.simplices_ = {
            dim: {
                "vertex_array": simplices[:-1].astype(int),
                "filtrations": simplices[-1],
            }
            for dim, simplices in enumerate(
                self._get_simplices()
            )
        }
        self.vprint("Done getting simplices.")
        simplex_tree_ = SimplexTree()
        self.vprint("Constructing simplex tree...")
        for dim in range(self.max_dimension + 2):
            simplex_tree_.insert_batch(**self.simplices_[dim])
        self.vprint("Done constructing simplex tree...")
        return simplex_tree_

    def _get_simplices(
        self,
    ):
        @jit(nopython=True, parallel=True)
        def _get_simplices_numba(dm, max_dimension):
            def choose_2(n):
                return n * (n - 1) // 2

            def choose_3(n):
                return n * (n - 1) * (n - 2) // 6
            num_vertices = dm.shape[0]
            num_edges = choose_2(num_vertices)
            num_faces = choose_3(num_vertices)
            arr_vertices = np.empty((2, num_vertices))
            arr_edges = np.empty((3, num_edges))
            arr_faces = np.empty((4, num_faces))
            for vertex_ix in prange(num_vertices):
                arr_vertices[0, vertex_ix] = vertex_ix
                arr_vertices[1, vertex_ix] = np.min(dm[vertex_ix])
                for vertex_jx in range(vertex_ix + 1, num_vertices):
                    edge_ix = choose_2(num_vertices) - 1 - (
                        choose_2(num_vertices - vertex_ix - 1)
                        + num_vertices - vertex_jx - 1
                    )
                    arr_edges[0, edge_ix] = vertex_ix
                    arr_edges[1, edge_ix] = vertex_jx
                    arr_edges[2, edge_ix] = np.min(
                        np.maximum(dm[vertex_ix], dm[vertex_jx])
                    )
                    if max_dimension > 0:
                        for vertex_kx in range(vertex_jx + 1, num_vertices):
                            face_ix = choose_3(num_vertices) - 1 - (
                                choose_3(num_vertices - vertex_ix - 1)
                                + choose_2(num_vertices - vertex_jx - 1)
                                + num_vertices - vertex_kx - 1
                            )
                            arr_faces[0, face_ix] = vertex_ix
                            arr_faces[1, face_ix] = vertex_jx
                            arr_faces[2, face_ix] = vertex_kx
                            arr_faces[3, face_ix] = np.min(
                                np.maximum(
                                    np.maximum(dm[vertex_ix], dm[vertex_jx]),
                                    dm[vertex_kx]
                                )
                            )
            return arr_vertices, arr_edges, arr_faces

        res = _get_simplices_numba(self._dm_, self.max_dimension)[
            : self.max_dimension + 2
        ]
        if self.max_filtration < np.inf:
            return (
                arr[:, arr[-1, :] <= self.max_filtration]
                for arr in res
            )
        else:
            return res

    def _format_persistence(
        self,
        persistence,
    ):
        if len(persistence) == 0:
            max_hom_dim = 0
        else:
            max_hom_dim = max([dim for dim, gen in persistence])
        persistence_formatted = [
            np.array([gen for dim, gen in persistence if dim == i]).reshape(
                -1, 2
            )
            for i in range(max_hom_dim + 1)
        ]
        persistence_sorted = [
            hom[
                np.argsort(
                    np.diff(hom, axis=1).reshape(
                        -1,
                    )
                )
            ]
            for hom in persistence_formatted
        ]
        while len(persistence_sorted) < self.max_dimension + 1:
            persistence_sorted.append(np.empty(shape=(0, 2)))
        return persistence_sorted

    def plot_persistence(
        self,
        **plotting_kwargs,
    ) -> gobj.Figure:
        """Method plotting the persistent homology of a Dowker complex.
        Underlying instance of `DowkerComplex` must be fitted and have the
        attribute `persistence_`.

        Args:
            plotting_kwargs (optional): Keyword arguments passed to the
                function `plotting.persistence_plotting.plot_persistences`,
                such as `marker_size`.

        Returns:
            `plotly.graph_objs.Figure`: A plot of the persistence diagram.
        """
        if not hasattr(self, "persistence_"):
            raise AttributeError(
                "This instance does not have the attribute `persistence_`. "
                "Run `fit_transform` before plotting."
            )
        fig = plot_persistences(
            [self.persistence_],
            **plotting_kwargs,
        )
        return fig

    def plot_points(
        self,
        indicate_witnesses: bool = True,
        use_colors: bool = True,
        **plotting_kwargs,
    ) -> gobj.Figure:
        """Method plotting the vertices and witnesses of a Dowker complex.
        Underlying instance of `DowkerComplex` must be fitted and have the
        attributes `vertices_` and `witnesses_`. Works for point clouds up to
        dimension three only.

        Args:
            indicate_witnesses (bool, optional): Whether or not to indicate the
                witness points by a cross as opposed to a dot.
                Defaults to `True`.
            use_colors (bool, optional): Whether or not to color the vertices
                and witnesses in different colors. Defaults to `True`.
            plotting_kwargs (optional): Keyword arguments passed to the
                function `plotting.point_cloud_plotting.plot_point_cloud`, such
                as `marker_size` and `colorscale`.

        Returns:
            `plotly.graph_objs.Figure`: A plot of the vertex and witness point
                clouds.
        """
        if not hasattr(self, "vertices_") and hasattr(self, "witnesses_"):
            raise AttributeError(
                "This instance does not have the attributes `vertices_` and "
                "`witnesses_`. Run `fit_transform` before plotting."
            )
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
            **plotting_kwargs,
        )

    def plot_skeleton(
        self,
        k: int = 2,
        threshold: float = np.inf,
        indicate_witnesses: bool = True,
        use_colors: bool = True,
        **plotting_kwargs,
    ) -> gobj.Figure:
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
            `plotly._figure.Figure`: A plot of the k-skeleton.
        """
        if self._points_.shape[1] not in {1, 2}:
            raise Exception(
                "Plotting of the skeleton is supported only "
                "for data sets of dimension at most 2."
            )
        if k not in {0, 1, 2}:
            raise Exception("The value of `k` must be either 0, 1 or 2.")
        check_is_fitted(self, attributes="complex_")
        complex = self.complex_.copy()
        complex.prune_above_filtration(threshold)
        vertex_ixs = [
            simplex[0]
            for simplex, filtration in complex.get_skeleton(dimension=1)
            if len(simplex) == 1
        ]
        points = np.concatenate(
            [self.vertices_[vertex_ixs].reshape(-1, 2), self.witnesses_]
        )
        labels = np.concatenate(
            [self._labels_vertices_[vertex_ixs], self._labels_witnesses_]
        )
        if k >= 1:
            lines = self._points_[
                [
                    simplex
                    for simplex, filtration in complex.get_skeleton(
                        dimension=1
                    )
                    if len(simplex) == 2
                ]
            ].reshape(-1, 2, 2)
        fig = plot_point_cloud(
            points,
            labels=labels,
            indicate_outliers=indicate_witnesses,
            indicate_labels=use_colors,
            colorscale="wong",
            lines=lines if k >= 1 else None,
            **plotting_kwargs,
        )
        if k == 2:
            two_simplices_with_filtration = [
                (spx, filtration)
                for spx, filtration in complex.get_skeleton(dimension=2)
                if len(spx) >= 3
            ]
            two_simplices = np.array(
                [
                    self.vertices_[two_simplex]
                    for two_simplex, filtration
                    in two_simplices_with_filtration
                ]
            )
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
                    line_color="grey",
                )
                fig.add_trace(polygon)
        fig_ref = plot_point_cloud(self._points_)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            xrange = fig_ref.full_figure_for_development().layout.xaxis.range
            yrange = fig_ref.full_figure_for_development().layout.yaxis.range
        fig.update_layout(xaxis_range=xrange, yaxis_range=yrange)
        return fig

    def plot_interactive_skeleton(
        self,
        k: int = 2,
        indicate_witnesses: bool = True,
        use_colors: bool = True,
        **plotting_kwargs,
    ) -> gobj.Figure:
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
            `plotly.graph_objs.Figure`: An interactive plot of the k-skeleton.
        """
        if self._points_.shape[1] not in {1, 2}:
            raise Exception(
                "Plotting of the skeleton is supported only "
                "for data sets of dimension at most 2."
            )
        if k not in {0, 1, 2}:
            raise Exception("The value of `k` must be either 0, 1 or 2.")
        fig_combined = self.plot_skeleton(
            threshold=0,
            k=0,
            indicate_witnesses=indicate_witnesses,
            use_colors=use_colors,
            **plotting_kwargs,
        )
        distances = np.concatenate(
            [
                [0],
                np.unique(
                    [
                        filtration
                        for simplex, filtration
                        in self.complex_.get_filtration()
                    ]
                ),
            ]
        )
        datum_ixs = defaultdict(list)
        datum_ix = len(fig_combined.data)
        for dist_ix, dist in enumerate(distances):
            fig = self.plot_skeleton(
                threshold=dist,
                k=k,
                indicate_witnesses=indicate_witnesses,
                use_colors=use_colors,
                **plotting_kwargs,
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
                label=str(np.round(distances[dist_ix], 6)),
            )
            for ix in datum_ixs[dist_ix]:
                step["args"][0]["visible"][ix] = True  # type: ignore
            steps.append(step)
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Threshold: "},
                pad={"t": 50},
                steps=steps,
            )
        ]
        fig_combined.update_layout(sliders=sliders)
        return fig_combined
