from typing import Optional

import numpy as np
import numpy.typing as npt
from gudhi import SimplexTree  # ty:ignore[unresolved-import]
from numba import jit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self


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
            homology. Defaults to `2`.
        metric (str, optional): The metric used to compute distance between
            data points. Must be one of the metrics listed in
            ``sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
            Defaults to `"euclidean"`.
        metric_params (dict, optional): Additional parameters to be passed to
            the distance function. Defaults to `None` (empty dict).
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
        metric_params: Optional[dict] = None,
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
        y: None = None,  # noqa: ARG002
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
        if len(X) != 2:
            raise ValueError(
                f"X must contain exactly 2 arrays (vertices and witnesses); "
                f"received {len(X)} arrays"
            )
        vertices, witnesses = X
        if vertices.ndim != 2 or witnesses.ndim != 2:
            raise ValueError(
                "Vertices and witnesses must be 2D arrays; "
                f"received vertices.ndim={vertices.ndim} and "
                f"witnesses.ndim={witnesses.ndim}"
            )
        if vertices.shape[1] != witnesses.shape[1]:
            raise ValueError(
                "The vertices and witnesses must be of the same "
                f"dimensionality; received dim(vertices)={vertices.shape[1]} "
                f"and dim(witnesses)={witnesses.shape[1]}"
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
        X: Optional[list[npt.NDArray]] = None,  # noqa: ARG002
    ) -> list[npt.NDArray[np.float64]]:
        """Method that transforms a `DowkerComplex`-instance fitted to a pair
        of point clouds consisting of vertices and witnesses by computing the
        persistent homology of the associated Dowker complex.

        Args:
            X (list[numpy.ndarray], optional): Not used; fitted data is used.
                Present for API consistency with scikit-learn.

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
        if min(len(self.vertices_), len(self.witnesses_)) == 0:
            self.persistence_ = [
                np.empty((0, 2), dtype=np.float64)
                for _ in range(self.max_dimension + 1)
            ]
        else:
            self.vprint("Computing persistent homology...")
            persistence_dim_max = (
                self.complex_.dimension() <= self.max_dimension
            )
            self.persistence_ = self._format_persistence(
                self.complex_.persistence(
                    homology_coeff_field=self.coeff,
                    min_persistence=0.0,
                    persistence_dim_max=persistence_dim_max,
                )
            )
            self.vprint("Done computing persistent homology.")
        return self.persistence_

    def _get_complex(self):
        self.metric_params = (
            self.metric_params if self.metric_params is not None else dict()
        )
        self._dm_ = pairwise_distances(
            self.vertices_,
            self.witnesses_,
            metric=self.metric,
            **self.metric_params,
        )
        self.vprint("Getting simplices...")
        self.simplices_ = {
            dim: {
                "vertex_array": simplices[:-1].astype(int),
                "filtrations": simplices[-1],
            }
            for dim, simplices in enumerate(self._get_simplices())
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
            for vertex_ix in prange(num_vertices):  # ty:ignore[not-iterable]
                arr_vertices[0, vertex_ix] = vertex_ix
                arr_vertices[1, vertex_ix] = np.min(dm[vertex_ix])
                for vertex_jx in range(vertex_ix + 1, num_vertices):
                    edge_ix = (
                        choose_2(num_vertices)
                        - 1
                        - (
                            choose_2(num_vertices - vertex_ix - 1)
                            + num_vertices
                            - vertex_jx
                            - 1
                        )
                    )
                    arr_edges[0, edge_ix] = vertex_ix
                    arr_edges[1, edge_ix] = vertex_jx
                    arr_edges[2, edge_ix] = np.min(
                        np.maximum(dm[vertex_ix], dm[vertex_jx])
                    )
                    if max_dimension > 0:
                        for vertex_kx in range(vertex_jx + 1, num_vertices):
                            face_ix = (
                                choose_3(num_vertices)
                                - 1
                                - (
                                    choose_3(num_vertices - vertex_ix - 1)
                                    + choose_2(num_vertices - vertex_jx - 1)
                                    + num_vertices
                                    - vertex_kx
                                    - 1
                                )
                            )
                            arr_faces[0, face_ix] = vertex_ix
                            arr_faces[1, face_ix] = vertex_jx
                            arr_faces[2, face_ix] = vertex_kx
                            arr_faces[3, face_ix] = np.min(
                                np.maximum(
                                    np.maximum(dm[vertex_ix], dm[vertex_jx]),
                                    dm[vertex_kx],
                                )
                            )
            return arr_vertices, arr_edges, arr_faces

        res = _get_simplices_numba(self._dm_, self.max_dimension)[
            : self.max_dimension + 2
        ]
        if self.max_filtration < np.inf:
            return (arr[:, arr[-1, :] <= self.max_filtration] for arr in res)
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
