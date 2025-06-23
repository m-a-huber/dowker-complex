import numpy as np
import plotly.graph_objs as gobj  # type: ignore
from gudhi import plot_persistence_barcode  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from . import cs_wong


def plot_persistences(
    persistences,
    homology_dimensions=None,
    without_infty=False,
    bandwidths=None,
    to_scale=False,
    marker_size=5.0,
    titles=None,
    display_plot=False,
    plotly_params=None,
):
    """Function plotting a collection of persistence diagrams. Inspired by and
    building upon code from the `giotto-tda` package [1].

    Args:
        persistences (list[list[numpy.ndarray]]):
            The data of the persistence diagrams to be plotted. The format of
            this data must be a list each of whose entries contains the data
            of an individual persistence diagram. This data, in turn, must be
            a list of NumPy-arrays of shape (n_generators, 2), where the i-th
            entry of the list is an array containing the birth and death times
            of the homological generators in dimension i-1. In particular,
            the list must start with 0-dimensional homology and contains
            information from consecutive homological dimensions.
        homology_dimensions (list[list[int] or None], optional): A list of the
            homological dimensions by which to filter each of the persistence
            diagrams. Each entry of this list should be a list of integers
            containing the dimensions of the persistence diagram to be plotted,
            or None. If such a list is None, then all homological dimensions
            present are taken into account for that persistence diagram.
            For instance, setting `homology_dimensions` to [[0], [1,2], None]
            will result in plotting the zeroth, the first and second, and the
            entire homology of the first, second and third persistence diagram,
            respectively. Passing None is equivalent to passing the list
            [None, ..., None]. Defaults to None.
        without_infty (bool, optional): Whether or not to plot a horizontal
            line corresponding to generators dying at infinity.
            Defaults to False.
        bandwidths (list(float), optional): A list of floats indicating the
            width of e.g. confidence bands to be plotted, where None results
            in no band being plotted. If `bandwidths` is set to e.g.
            [1, None, 3], the first and third persistence diagram will be
            endowed with a dashed line at distance 1 and 3 from the diagonal,
            respectively, whereas the second diagram will remain unchanged.
            Passing None is equivalent to passing the list [None, ..., None].
            Defaults to None.
        to_scale (bool, optional): Whether or not to use the same scale across
            all axes of the plot. Defaults to False.
        marker_size (float, optional): The size of the markers used to plot
            homological generators. Defaults to 5.0.
        titles (list[str], optional): A list of strings containing the titles
            of the persistence diagrams to be plotted, where None results in
            no title being plotted. Each title will be displayed above the
            corresponding diagram. If `titles` is set to e.g. ["A", None, "C"],
            the first and third persistence diagram will be endowed with the
            titles "A" and "C", respectively, whereas the second diagram will
            remain unchanged. Passing None is equivalent to passing the list
            [None, ..., None]. Defaults to None.
        display_plot (bool, optional): Whether or not to call show() on
            the resulting Plotly figure, as opposed to just returning
            the figure. Defaults to False.
        plotly_params (dict, optional): Custom parameters to configure the
            plotly figure. Allowed keys are ``"trace"`` and ``"layout"``, and
            the corresponding values should be dictionaries containing keyword
            arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`. Defaults to None.

    Returns:
        :class:`plotly.graph_objs._figure.Figure`: Figure containing the
            plotted persistence diagrams.

    References:
        [1]: giotto-tda: A Topological Data Analysis Toolkit for Machine
            Learning and Data Exploration, Tauzin et al, J. Mach. Learn. Res.
            22.39 (2021): 1-6.
    """
    # Validate `persistences` parameter
    if not isinstance(persistences, list):
        raise ValueError(
            "The `persistences` parameter must be a list."
        )
    if not all(
        isinstance(persistence, list)
        for persistence in persistences
    ):
        raise ValueError(
            "Each entry of the `persistences` parameter must be a list."
        )
    if not all(
        isinstance(dim, np.ndarray)
        for persistence in persistences
        for dim in persistence
    ):
        raise ValueError(
            "Each entry of each entry of the `persistences` "
            "parameter must be a NumPy-array."
        )
    if not all(
        dim.ndim == 2
        and dim.shape[1] == 2
        for persistence in persistences
        for dim in persistence
    ):
        raise ValueError(
            "Each entry of each entry of the `persistences` "
            "parameter must be a NumPy-array of shape (n_generators, 2)."
        )
    # Validate `homology_dimensions` parameter
    if not isinstance(homology_dimensions, (list, type(None))):
        raise ValueError(
            "The `homology_dimensions` parameter must be a list or None."
        )
    if homology_dimensions is not None:
        if not all(
            isinstance(homology_dimension, (list, type(None)))
            for homology_dimension in homology_dimensions
        ):
            raise ValueError(
                "Each entry of the `homology_dimensions` "
                "parameter must be a list or None."
            )
        if not all(
            isinstance(dim, int)
            and dim >= 0
            and dim <= len(persistences[i]) - 1
            for i, homology_dimension in enumerate(homology_dimensions)
            if homology_dimension is not None
            for dim in homology_dimension
        ):
            raise ValueError(
                "Each entry of each entry of the `homology_dimensions` "
                "parameter must be an integer between zero and the maximal "
                "homological dimension of the corresponding diagram."
            )
    # Validate `without_infty` parameter
    if not isinstance(without_infty, bool):
        raise ValueError(
            "The `without_infty` parameter must be a Boolean."
        )
    # Validate `bandwidths` parameter
    if not isinstance(bandwidths, (list, type(None))):
        raise ValueError(
            "The `bandwidths` parameter must be a list or None."
        )
        if bandwidths is not None:
            if not (
                all(
                    isinstance(bandwidth, (int, float, type(None)))
                    for bandwidth in bandwidths
                )
                and all(
                    bandwidth is None or (
                        bandwidth >= 0 and bandwidth < float("inf")
                    )
                    for bandwidth in bandwidths
                )
            ):
                raise ValueError(
                    "Each entry of the `bandwidths` parameter must "
                    "be a non-negative finite number or None."
                )
    # Validate `to_scale` parameter
    if not isinstance(to_scale, bool):
        raise ValueError(
            "The `to_scale` parameter must be a Boolean."
        )
    # Validate `marker_size` parameter
    if not (
        isinstance(marker_size, (int, float))
        and marker_size > 0
        and marker_size < float("inf")
    ):
        raise ValueError(
            "The `marker_size` parameter must be a finite positive number."
        )
    # Validate `titles` parameter
    if not isinstance(titles, (list, type(None))):
        raise ValueError(
            "The `titles` parameter must be a list or None."
        )
    if titles is not None:
        if not all(
            isinstance(el, (str, type(None)))
            for el in titles
        ):
            raise ValueError(
                "Each entry of the `titles` parameter "
                "must be a string or None."
            )
    # Validate `display_plot` parameter
    if not isinstance(display_plot, bool):
        raise ValueError(
            "The `display_plot` parameter must be a Boolean."
        )
    # Validate `plotly_params` parameter
    if not isinstance(plotly_params, (dict, type(None))):
        raise ValueError(
            "The `plotly_params` parameter must "
            "be either a dictionary or None."
        )

    diagrams = persistences
    n_diagrams = len(diagrams)
    dgms = tuple(
        _make_plottable(dgm, for_gudhi=False, without_infty=without_infty)
        for dgm in diagrams
    )
    dgms_indexed = tuple(
        np.c_[dgm, i * np.ones(dgm.shape[0])]
        for i, dgm in enumerate(dgms)
    )
    dgms_combined = np.vstack(dgms_indexed)
    if homology_dimensions is None:
        homology_dimensions = [
            np.unique(dgms_combined[dgms_combined[:, -1] == i][:, 2])
            for i, dgm in enumerate(diagrams)
        ]
    else:
        homology_dimensions = [
            np.unique(dgms_combined[dgms_combined[:, -1] == i][:, 2])
            if homology_dimensions[i] is None
            else homology_dimensions[i]
            for i, dgm in enumerate(diagrams)
        ]

    min_xval_display_list = []
    max_xval_display_list = []
    min_yval_display_list = []
    max_yval_display_list = []
    posinfinity_val_list = []
    neginfinity_val_list = []

    for i, dgm in enumerate(diagrams):
        diagram = dgms_combined[dgms_combined[:, -1] == i][:, :-1]
        diagram_no_dims = diagram[:, :2]
        posinfinite_mask = np.isposinf(diagram_no_dims)
        neginfinite_mask = np.isneginf(diagram_no_dims)
        # Catch case where persistence is zero
        if len(diagram_no_dims) == 0:
            min_xval = 0
            max_xval = 0
            min_yval = 0
            max_yval = 0
        else:
            min_xval = np.min(
                np.where(neginfinite_mask, np.inf, diagram_no_dims)[:, 0]
            )
            max_xval = np.max(
                np.where(posinfinite_mask, -np.inf, diagram_no_dims)[:, 0]
            )
            min_yval = np.min(
                np.where(neginfinite_mask, np.inf, diagram_no_dims)[:, 1]
            )
            max_yval = np.max(
                np.where(posinfinite_mask, -np.inf, diagram_no_dims)[:, 1]
            )
        xparameter_range = max(max_xval - min_xval, 1)
        yparameter_range = max(max_yval - min_yval, 1)
        extra_space_factor = 0.05
        has_posinfinite_death = np.any(posinfinite_mask[:, 1])
        if has_posinfinite_death:
            posinfinity_val = max(
                max_xval + 0.05 * xparameter_range,
                max_yval + 0.05 * yparameter_range
            )
            posinfinity_val_list.append(posinfinity_val)
            extra_space_factor += 0.05
        else:
            posinfinity_val_list.append(None)
        has_neginfinite_death = np.any(neginfinite_mask[:, 1])
        if has_neginfinite_death:
            neginfinity_val = min(
                min_xval - 0.05 * xparameter_range,
                min_yval - 0.05 * yparameter_range
            )
            neginfinity_val_list.append(neginfinity_val)
            extra_space_factor += 0.05
        else:
            neginfinity_val_list.append(None)
        extra_xspace = extra_space_factor * xparameter_range
        extra_yspace = extra_space_factor * yparameter_range
        min_xval_display = min_xval - extra_xspace
        max_xval_display = max_xval + extra_xspace
        min_yval_display = min_yval - extra_yspace
        max_yval_display = max_yval + extra_yspace
        min_xval_display_list.append(min_xval_display)
        max_xval_display_list.append(max_xval_display)
        min_yval_display_list.append(min_yval_display)
        max_yval_display_list.append(max_yval_display)

    min_xval_display = min(min_xval_display_list)
    max_xval_display = max(max_xval_display_list)
    min_yval_display = min(
        min(min_yval_display_list),
        min_xval_display
    )
    max_yval_display = max(
        max(max_yval_display_list),
        max_xval_display
    )
    if any(
        posinfinity_val is not None
        for posinfinity_val in posinfinity_val_list
    ):
        posinfinity_val_display = max(
            posinfinity_val
            for posinfinity_val in posinfinity_val_list
            if posinfinity_val is not None
        )
    if any(
        neginfinity_val is not None
        for neginfinity_val in neginfinity_val_list
    ):
        neginfinity_val_display = min(
            neginfinity_val
            for neginfinity_val in neginfinity_val_list
            if neginfinity_val is not None
        )
    if bandwidths is not None:
        bandwidths_ = [
            bandwidth
            if bandwidth is not None else 0
            for bandwidth in bandwidths
        ]
        min_yval_display = np.min(
            [min_yval_display, min_xval_display + np.min(bandwidths_)]
        )
        max_yval_display = np.max(
            [max_yval_display, max_xval_display + np.max(bandwidths_)]
        )
        yparameter_range = max_yval_display - min_yval_display
        extra_space_factor = 0.05
        extra_yspace = extra_space_factor * yparameter_range
        if any(
            posinfinity_val is not None
            for posinfinity_val in posinfinity_val_list
        ):
            posinfinity_val_display = np.max([
                posinfinity_val_display,
                max_xval_display + np.max(bandwidths_)
            ])
            max_yval_display += extra_yspace
        if any(
            neginfinity_val is not None
            for neginfinity_val in neginfinity_val_list
        ):
            neginfinity_val_display = np.min([
                neginfinity_val_display,
                min_xval_display + np.min(bandwidths_)
            ])
            min_yval_display -= extra_yspace

    rows = n_diagrams // 2 + n_diagrams % 2
    cols = min(n_diagrams, 2)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    fig.update_annotations(yshift=25)

    for i, dgm in enumerate(diagrams):
        # Add diagonal y=x
        fig.add_trace(
            gobj.Scatter(
                x=[min_xval_display, max_xval_display],
                y=[min_xval_display, max_xval_display],
                mode="lines",
                line={"width": 1, "color": "black"},
                showlegend=False,
                hoverinfo="none",
            ),
            row=(i // 2) + 1,
            col=(i % 2) + 1,
        )

        diagram = dgms_combined[dgms_combined[:, -1] == i][:, :-1]
        # Add homological generators
        for dim in homology_dimensions[i]:
            if dim != np.inf:
                name = f"H{int(dim)}"
            else:
                name = "Any homology dimension"
            subdiagram = diagram[diagram[:, 2] == dim]
            unique, inverse, counts = np.unique(
                subdiagram, axis=0, return_inverse=True, return_counts=True
            )
            hovertext = [
                f"{tuple(unique[unique_row_index][:2])}"
                + (
                    f", multiplicity: {counts[unique_row_index]}"
                    if counts[unique_row_index] > 1
                    else ""
                )
                for unique_row_index in inverse
            ]
            y = subdiagram[:, 1]
            # if len(posinfinity_val_list) > i:
            if posinfinity_val_list[i] is not None:
                y[np.isposinf(y)] = posinfinity_val_display
            # if len(neginfinity_val_list) > i:
            if neginfinity_val_list[i] is not None:
                y[np.isneginf(y)] = neginfinity_val_display
            fig.add_trace(
                gobj.Scatter(
                    x=subdiagram[:, 0],
                    y=y,
                    mode="markers",
                    hoverinfo="text",
                    hovertext=hovertext,
                    name=name,
                    marker=dict(
                        color=f"rgb{cs_wong.rgbs[int(dim)]}",
                        size=marker_size,
                    ),
                    showlegend=False,
                ),
                row=(i // 2) + 1,
                col=(i % 2) + 1,
            )

    for i, dgm in enumerate(diagrams):
        # Update x-axes
        fig.update_xaxes(
            {
                "title": "Birth",
                "side": "bottom",
                "type": "linear",
                "range": [min_xval_display, max_xval_display],
                "constrain": "domain",
                "autorange": False,
                "ticks": "outside",
                "showline": True,
                "zeroline": True,
                "linewidth": 1,
                "linecolor": "black",
                "mirror": False,
                "showexponent": "all",
                "exponentformat": "e",
            },
            row=(i // 2) + 1,
            col=(i % 2) + 1,
        )
        # Update y-axes
        fig.update_yaxes(
            {
                "title": "Death",
                "side": "left",
                "type": "linear",
                "range": [min_yval_display, max_yval_display],
                "constrain": "domain",
                "autorange": False,
                "ticks": "outside",
                "showline": True,
                "zeroline": True,
                "linewidth": 1,
                "linecolor": "black",
                "mirror": False,
                "showexponent": "all",
                "exponentformat": "e",
            },
            row=(i // 2) + 1,
            col=(i % 2) + 1,
        )
        if to_scale:
            fig.update_xaxes(
                {
                    "scaleanchor": "x",
                    "scaleratio": 1,
                },
                row=(i // 2) + 1,
                col=(i % 2) + 1,
            )
            fig.update_yaxes(
                {
                    "scaleanchor": "x",
                    "scaleratio": 1,
                },
                row=(i // 2) + 1,
                col=(i % 2) + 1,
            )

    fig.update_layout(
            width=500 * cols,
            height=500 * rows,
            plot_bgcolor="white"
    )

    # Add a horizontal dashed line for points with infinite death
    for i, dgm in enumerate(diagrams):
        if posinfinity_val_list[i] is not None:
            fig.add_trace(
                gobj.Scatter(
                    x=[min_xval_display, max_xval_display],
                    y=[posinfinity_val_display, posinfinity_val_display],
                    mode="lines",
                    line={"dash": "dash", "width": 0.5, "color": "black"},
                    showlegend=False,
                    hoverinfo="none",
                ),
                row=(i // 2) + 1,
                col=(i % 2) + 1,
            )

    # Add a horizontal dashed line for points with negative death
    for i, dgm in enumerate(diagrams):
        if neginfinity_val_list[i] is not None:
            fig.add_trace(
                gobj.Scatter(
                    x=[min_xval_display, max_xval_display],
                    y=[neginfinity_val_display, neginfinity_val_display],
                    mode="lines",
                    line={"dash": "dash", "width": 0.5, "color": "black"},
                    showlegend=False,
                    hoverinfo="none",
                ),
                row=(i // 2) + 1,
                col=(i % 2) + 1,
            )

    # Add dashed line indicating bandwidths
    for i, dgm in enumerate(diagrams):
        if bandwidths is not None and bandwidths[i] is not None:
            bandwidth = bandwidths[i]
            fig.add_trace(
                gobj.Scatter(
                    x=[
                        min_xval_display,
                        max_xval_display
                    ],
                    y=[
                        min_xval_display + bandwidth,
                        max_xval_display + bandwidth
                    ],
                    mode="lines",
                    line={"dash": "dot", "width": 1, "color": "black"},
                    showlegend=False,
                    hoverinfo="none",
                ),
                row=(i // 2) + 1,
                col=(i % 2) + 1,
            )

    # Add the legend shared across both plots
    dims = np.unique(np.concatenate([arr for arr in homology_dimensions]))
    for dim in dims:
        fig.add_trace(
            gobj.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                showlegend=True,
                name=f"H{int(dim)}",
                marker=dict(color=f"rgb{cs_wong.rgbs[int(dim)]}"),
            )
        )

    # Add legend for diagonal
    fig.add_trace(
        gobj.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"width": 1, "color": "black"},
            showlegend=True,
            name="y=x",
        )
    )

    # Add legend for infinity bars
    if not without_infty:
        fig.add_trace(
            gobj.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"dash": "dash", "width": 1, "color": "black"},
                showlegend=True,
                name="y=\u00b1\u221e",
            )
        )

    # Add legend for bandwidths
    if bandwidths is not None:
        if any(bandwidth is not None for bandwidth in bandwidths):
            name = (
                "Confidence band" if len(diagrams) == 1 else "Confidence bands"
            )
            fig.add_trace(
                gobj.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line={"dash": "dot", "width": 1, "color": "black"},
                    showlegend=True,
                    name=name,
                )
            )

    # Update traces and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))
    if display_plot:
        fig.show()
    return fig


def plot_barcodes(persistences, without_infty=False):
    """Function plotting a collection of persistence barcodes.

    Args:
        persistences (list[list[numpy.ndarray]]):
            The data of the persistence diagrams to be plotted. The format of
            this data must be a list each of whose entries contains the data
            of an individual persistence diagram. This data, in turn, must be
            a list of NumPy-arrays of shape (n_generators, 2), where the i-th
            entry of the list is an array containing the birth and death
            times of the homological generators in dimension i-1. In
            particular, the list must start with 0-dimensional homology and
            contains information from consecutive homological dimensions.
        without_infty (bool, optional): Whether or not to plot a horizontal
            line corresponding to generators dying at infinity.
            Defaults to False.

    Returns:
        tuple(matplotlib.axes._axes.Axes): Tuple containing the plotted
            persistence barcodes.
    """
    # Validate `persistences` parameter
    if not isinstance(persistences, list):
        raise ValueError(
            "The `persistences` parameter must be a list."
        )
    if not all(
        isinstance(persistences, list)
        for persistence in persistences
    ):
        raise ValueError(
            "Each entry of the `persistences` parameter must be a list."
        )
    if not all(
        isinstance(dim, np.ndarray)
        for persistence in persistences
        for dim in persistence
    ):
        raise ValueError(
            "Each entry of each entry of the `persistences` "
            "parameter must be a NumPy-array."
        )
    if not all(
        dim.ndim == 2
        and dim.shape[1] == 2
        for persistence in persistences
        for dim in persistence
    ):
        raise ValueError(
            "Each entry of each entry of the `persistences` "
            "parameter must be a NumPy-array of shape (n_generators, 2)."
        )
    # Validate `without_infty` parameter
    if not isinstance(without_infty, bool):
        raise ValueError(
            "The `without_infty` parameter must be a Boolean."
        )

    data = tuple(
        _make_plottable(
            persistences[i],
            for_gudhi=True,
            without_infty=without_infty
        )
        for i in range(len(persistences))
    )
    figs = tuple(
        plot_persistence_barcode(data[i])
        for i in range(len(persistences))
    )
    return figs


def format_gudhi_persistence(persistence, sort_by_lifetime=False):
    """Function converting persistence data given in the format used in the
    GUDHI library into the format required by the function `plot_persistences`.

    Args:
        persistence (list[tuple]): Persistence data given in the format used
            in the GUDHI library. That is, a list where each entry is a tuple
            of the format (dim, (birth, death)) where `dim`, `birth` and
            `death` denote the homological dimension and the birth and death
            times of a homological generator, respectively.
        sort_by_lifetime (bool, optional): Whether or not to sort the
            homological generators in the output by increasing lifetime
            (within each homological dimension). Defaults to False.

    Returns:
        list[numpy.ndarray]: A list containing the same persistence data as
            `persistence` in the same format as that of an entry of a list
            that is a suitable input to the function `plot_persistences` as
            described there.
    """
    aux_array = np.array([
        [dim, *birth_death_pair]
        for dim, birth_death_pair in persistence
    ])
    dims = np.unique(aux_array[:, 0])
    res = [
        aux_array[aux_array[:, 0] == dim][:, 1:]
        for dim in dims
    ]
    if sort_by_lifetime:

        def _sort_by_lifetime(arr):
            return arr[
                np.argsort(
                    np.diff(arr, axis=1).reshape(
                        -1,
                    )
                )
            ]

        res = list(map(_sort_by_lifetime, res))
    return res


def _make_plottable(persistence, for_gudhi, without_infty):
    if for_gudhi:
        if without_infty:
            return np.array(
                [
                    (ix, row)
                    for ix, arr in enumerate(persistence)
                    for row in arr
                    if row[-1] != np.inf
                ],
                dtype=object,
            )
        return np.array(
            [(ix, row) for ix, arr in enumerate(persistence) for row in arr],
            dtype=object,
        )
    res = np.vstack(
        [np.c_[
            arr, ix * np.ones(arr.shape[0])]
            for ix, arr in enumerate(persistence)
        ]
    )
    if without_infty:
        res = res[np.isfinite(res).all(axis=1)]
    return res
