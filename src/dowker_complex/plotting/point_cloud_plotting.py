import warnings

import numpy as np
import plotly.express as px  # type: ignore
import plotly.graph_objs as gobj  # type: ignore

from . import cs_wong


def plot_point_cloud(
    point_cloud,
    labels=None,
    dimension=None,
    names=None,
    marker_size=4.0,
    indicate_outliers=True,
    indicate_labels=True,
    opacity=0.8,
    to_scale=False,
    colorscale=None,
    with_colorbar=False,
    title=None,
    radius=None,
    radius_kwargs=None,
    lines=None,
    line_width=1.0,
    arrows=None,
    arrow_width=3.0,
    display_plot=False,
    plotly_params=None,
):
    """Function plotting the first 2 or 3 coordinates of a point cloud.
    Note: this function does not work on 1D arrays. Inspired by and building
    upon code from the `giotto-tda` package [1].

    Args:
        point_cloud (numpy.ndarray of shape (n_samples, n_dimensions)): Data
            points to be represented in a 2D or 3D scatter plot. Only the
            first 2 or 3 dimensions will be considered for plotting.
        labels (numpy.ndarray of shape (n_samples,), optional): Array of
            labels of data points that, if provided, are used to color-code
            the data points. Defaults to None.
        dimension (int, optional): Sets the dimension of the resulting plot.
            If ``None``, the dimension will be chosen between 2 and 3
            depending on the shape of `point_cloud`. Defaults to None.
        names (dict of int: str, optional): Dictionary translating each
            numeric label into a string representing its name. Should be of
            the format {label[int] : name[str]}. If provided, a legend will be
            added to the plot. Defaults to None.
        marker_size (float, optional): Sets the size of the markers in the
            plot. Must be a positive number. Defaults to 4.0.
        indicate_outliers (bool, optional): Whether or not to use a
            distinguished marker for points carrying a negative label. Useful
            for indicating points labelled as noise by a clustering algorithm.
            Ignored if no labels are provided. Defaults to True.
        indicate_labels (bool, optional): Whether or not to use colors to
            indicate the labels of the data points. Ignored if no labels are
            provided. Defaults to True.
        opacity (float, optional): Sets the opacity of the markers in the plot.
            Must be a number between 0 and 1. Defaults to 0.8.
        to_scale (bool, optional): Whether or not to use the same scale across
            all axes of the plot. Defaults to False.
        colorscale (str, optional): Which colorscale to use. Must be `"wong"`
            or a colorscale in plotly.express.colors.named_colorscales(). If
            `"wong"`, the colorscale described in [2] will be used, provided
            that the set of labels is a set of integers such that the
            difference between the maximum and the minimum label values is at
            most seven; otherwise the choice of `"wong"` will be ignored.
            Defaults to None.
        with_colorbar (bool, optional): Whether or not to display the colorbar
            corresponding to the chosen colorscale. Defaults to False.
        title (str, optional): The title of the plot, which will be displayed
            above it. If None, no title is created. Defaults to None.
        radius (float or numpy.ndarray of shape (n_samples,)): If not None,
            will create circles with desired radius around each data point. If
            a float, all circles will have that number as their radius. If an
            array of floats, circles of these radii will be created around each
            data point, respecting the order. Available for 2D plots only.
            Defaults to None.
        radius_kwargs (dict, optional): Additional parameters to be used in the
            creation of the circles around the data points such as `"opacity"`,
            `"fillcolor"` and `"line_color"`. These latter parameters default
            to 0.2, `"red"` and `"red"`, respectively. Defaults to None.
        lines (numpy.ndarray of shape (n_lines, 2, 2), optional): Array of
            data points to be connected with an line. Each entry should be an
            array of shape (2,2), consisting of the start and end point of the
            respective line. Available for 2D plots only. Defaults to None.
        line_width (float, optional): Thickness of the lines. Should be a
            positive number and only be provided if `lines` is not None.
            Defaults to 3.0.
        arrows (numpy.ndarray of shape (n_arrows, 2, 2), optional): Array of
            data points to be connected with an arrow. Each entry should be an
            array of shape (2,2), consisting of the start and end point of the
            respective arrow. Available for 2D plots only. Defaults to None.
        arrow_width (float, optional): Thickness of the arrows. Should be a
            positive number and only be provided if `arrows` is not None.
            Defaults to 3.0.
        display_plot (bool, optional): Whether or not to display the plot.
            Defaults to False.
        plotly_params (dict, optional): Custom parameters to configure the
            plotly figure. Allowed keys are ``"trace"`` and ``"layout"``, and
            the corresponding values should be dictionaries containing keyword
            arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`. Defaults to None.

    Returns:
        :class:`plotly.graph_objs._figure.Figure`: Figure representing a point
            cloud in 2-dim. or 3-dim. space.

    References:
        [1]: giotto-tda: A Topological Data Analysis Toolkit for Machine
            Learning and Data Exploration, Tauzin et al, J. Mach. Learn. Res.
            22.39 (2021): 1-6.
        [2]: https://www.nature.com/articles/nmeth.1618
    """
    # Validate `point_cloud` parameter
    try:
        point_cloud = np.asarray(point_cloud, dtype=float)
    except Exception as exc:
        raise ValueError(
            "The `point_cloud` parameter must be convertible into a "
            f"NumPy-array of floats, However, NumPy says: '{exc}'."
        )
    if not point_cloud.ndim == 2:
        raise ValueError(
            "The `point_cloud` parameter must be 2-dimensional."
        )
    if not point_cloud.shape[1] > 1:
        raise ValueError(
            "The `point_cloud` parameter must describe a "
            "point cloud of dimension 2 or higher."
        )
    # Validate `dimension` parameter
    if not isinstance(dimension, (int, type(None))):
        raise ValueError(
            "The `dimension` parameter must be either an integer or None."
        )
    if dimension is None:
        dimension = int(np.min((3, point_cloud.shape[1])))
    # Validate `labels` parameter; if no labels were provided, assign label 0
    # to each data point, and record if there were user-provided labels to use
    # in `names`.
    labels_were_provided = labels is not None
    if not labels_were_provided:
        labels = np.zeros(point_cloud.shape[0])
    else:
        try:
            labels = np.asarray(labels, dtype=float)
            labels = labels.reshape(point_cloud.shape[0])
        except Exception as exc:
            raise ValueError(
                "The `labels` parameter must be convertible into a "
                "NumPy-array of floats, and be of the same length as the "
                f"`point_cloud` parameter. However, NumPy says: '{exc}'."
            )
    # Validate `names` parameter
    if not isinstance(names, (dict, type(None))):
        raise ValueError(
            "The `names` parameter must be either a dictionary or None."
        )
    if names is not None:
        if not (
            all(isinstance(key, int) for key in names)
            and all(isinstance(names[key], str) for key in names)
        ):
            raise ValueError(
                "The `names` parameter must be a dictionary whose keys and "
                "values are integers and strings, respectively."
            )
        if not labels_were_provided:
            raise ValueError("No labels were provided.")
        if not np.array([
            label in names.keys()
            for label in np.unique(labels)
        ]).all():
            raise ValueError(
                "One or more labels are "
                "lacking a corresponding name."
            )
        if not np.array([
            isinstance(value, str)
            for value in names.values()
        ]).all():
            raise ValueError("All values of `names` should be strings.")
    # Validate `marker_size` parameter
    if not (
        isinstance(marker_size, (int, float))
        and marker_size > 0
        and marker_size < float("inf")
    ):
        raise ValueError(
            "The `marker_size` parameter must be a finite positive number."
        )
    # Validate `indicate_outliers` parameter
    if not isinstance(indicate_outliers, bool):
        raise ValueError(
            "The `indicate_outliers` parameter must be a Boolean."
        )
    # Validate `opacity` parameter
    if not (
        isinstance(opacity, (int, float))
        and opacity > 0
        and opacity <= 1
    ):
        raise ValueError(
            "The `opacity` parameter must be a number in (0, 1]."
        )
    # Validate `to_scale` parameter
    if not isinstance(to_scale, bool):
        raise ValueError(
            "The `to_scale` parameter must be a Boolean."
        )
    # Validate `colorscale` parameter
    if not isinstance(colorscale, (str, type(None))):
        raise ValueError(
            "The `colorscale` parameter must "
            "be either a string or None."
        )
        if (
            colorscale is not None
            and colorscale not in px.colors.named_colorscales()
        ):
            raise ValueError(
                "The `colorscale` parameter must be either None or a string "
                "representing a named Plotly-colorscale, i.e. a string "
                "contained in `px.colors.named_colorscales()`."
            )
    # Validate `with_colorbar` parameter
    if not isinstance(with_colorbar, bool):
        raise ValueError(
            "The `with_colorbar` parameter must be a Boolean."
        )
    # Validate `title` parameter
    if not isinstance(title, (str, type(None))):
        raise ValueError(
            "The `title` parameter must be a string or None."
        )
    # Validate `radius` parameter
    if radius is not None:
        if dimension != 2:
            raise ValueError(
                "The radius feature is available for "
                "2-dimensional plots only."
            )
        try:
            if isinstance(radius, float) or isinstance(radius, int):
                radius = float(radius)
                radius = radius * np.ones(point_cloud.shape[0], dtype=float)
            radius = np.array(radius, dtype=float)
        except Exception as exc:
            raise ValueError(
                "The `radius` parameter must be either None, a float or "
                "convertible into a NumPy-array of floats. However, NumPy "
                f"says: '{exc}'."
            )
        if not (
            radius.ndim == 1
            and radius.shape == (point_cloud.shape[0],)
        ):
            raise ValueError(
                "The `radius` parameter must be either None, a float or "
                "convertible into a NumPy-array of floats of shape "
                "(n_samples,)."
            )
    # Validate `radius_kwargs` parameter
    if radius_kwargs is not None:
        if dimension != 2:
            raise ValueError(
                "The radius feature is available for "
                "2-dimensional plots only."
            )
    elif radius is not None:
        radius_kwargs = dict()
    if radius is not None:
        radius_kwargs["opacity"] = radius_kwargs.get("opacity", 0.2)
        radius_kwargs["fillcolor"] = radius_kwargs.get("fillcolor", "red")
        radius_kwargs["line_color"] = radius_kwargs.get("line_color", "red")
    # Validate `lines` parameter
    if lines is not None:
        if dimension != 2:
            raise ValueError(
                "The line feature is available "
                "for 2-dimensional plots only."
            )
        try:
            lines = np.asarray(lines, dtype=float)
        except Exception as exc:
            raise ValueError(
                "The `lines` parameter must be either None or convertible "
                f"into a NumPy-array of floats. However, NumPy says: '{exc}'."
            )
        if not (
            lines.ndim == 3
            and lines.shape[1:] == (2, 2)
        ):
            raise ValueError(
                "The `lines` parameter must be either None or convertible "
                "into a NumPy-array of floats of shape (n_lines, 2, 2)."
            )
    # Validate `line_width` parameter
    if not (
        isinstance(line_width, (int, float))
        and line_width > 0
        and line_width < float("inf")
    ):
        raise ValueError(
            "The `line_width` parameter must be a finite positive number."
        )
    # Validate `arrows` parameter
    if arrows is not None:
        if dimension != 2:
            raise ValueError(
                "The arrow feature is available for 2-dimensional plots only."
            )
        try:
            arrows = np.asarray(arrows, dtype=float)
        except Exception as exc:
            raise ValueError(
                "The `arrows` parameter must be either None or convertible "
                f"into a NumPy-array of floats. However, NumPy says: '{exc}'."
            )
        if not (
            arrows.ndim == 3
            and arrows.shape[1:] == (2, 2)
        ):
            raise ValueError(
                "The `arrows` parameter must be either None or convertible "
                "into a NumPy-array of floats of shape (n_arrows, 2, 2)."
            )
    # Validate `arrow_width` parameter
    if not (
        isinstance(arrow_width, (int, float))
        and arrow_width > 0
        and arrow_width < float("inf")
    ):
        raise ValueError(
            "The `arrow_width` parameter must be a finite positive number."
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

    # Check consistency between `point_cloud` and `dimension`
    if point_cloud.shape[1] < dimension:
        raise ValueError(
            "Not enough dimensions available "
            "in the input point cloud."
        )

    # Check consistency between `names` and `with_colorbar`
    if min(names is not None, with_colorbar):
        raise ValueError(
            "The names and the colorbar cannot be displayed simultaneously."
        )

    if dimension == 2:
        layout = {
            "width": 600,
            "height": 600,
            "xaxis1": {
                "title": "0th",
                "side": "bottom",
                "type": "linear",
                "ticks": "outside",
                "anchor": "x1",
                "showline": True,
                "zeroline": True,
                "showexponent": "all",
                "exponentformat": "e",
            },
            "yaxis1": {
                "title": "1st",
                "side": "left",
                "type": "linear",
                "ticks": "outside",
                "anchor": "y1",
                "showline": True,
                "zeroline": True,
                "showexponent": "all",
                "exponentformat": "e",
            },
            "plot_bgcolor": "white",
        }

        fig = gobj.Figure(layout=layout)
        fig.update_xaxes(
            zeroline=True,
            linewidth=1,
            linecolor="black",
            mirror=False,
        )
        fig.update_yaxes(
            zeroline=True,
            linewidth=1,
            linecolor="black",
            mirror=False,
        )

        if names is None:
            if indicate_outliers:
                symbols = np.where(labels < 0, "cross", "circle")
            else:
                symbols = "circle"
            fig.add_trace(
                gobj.Scatter(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    mode="markers",
                    showlegend=False,
                    marker={
                        "size": marker_size,
                        "symbol": symbols,
                        "color": labels if indicate_labels else None,
                        "colorscale": "Viridis",
                        "opacity": opacity,
                    },
                )
            )
        else:
            for label in np.unique(labels):
                if indicate_outliers:
                    symbol = (lambda x: "cross" if x < 0 else "circle")(label)
                else:
                    symbol = "circle"
                fig.add_trace(
                    gobj.Scatter(
                        x=point_cloud[labels == label][:, 0],
                        y=point_cloud[labels == label][:, 1],
                        mode="markers",
                        name=names[label],
                        marker={
                            "size": marker_size,
                            "symbol": symbol,
                            "color": label if indicate_labels else None,
                            "colorscale": "Viridis",
                            "opacity": opacity,
                        },
                    )
                )
        if to_scale:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

    elif dimension == 3:
        scene = {
            "xaxis": {
                "title": "0th",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e",
            },
            "yaxis": {
                "title": "1st",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e",
            },
            "zaxis": {
                "title": "2nd",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e",
            },
        }

        fig = gobj.Figure()
        fig.update_layout(scene=scene)

        if names is None:
            if indicate_outliers:
                symbols = np.where(labels < 0, "cross", "circle")
            else:
                symbols = "circle"
            fig.add_trace(
                gobj.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode="markers",
                    showlegend=False,
                    marker={
                        "size": marker_size,
                        "symbol": symbols,
                        "color": labels if indicate_labels else None,
                        "colorscale": "Viridis",
                        "opacity": opacity,
                    },
                )
            )
        else:
            for label in np.unique(labels):
                if indicate_outliers:
                    symbol = (lambda x: "cross" if x < 0 else "circle")(label)
                else:
                    symbol = "circle"
            for label in np.unique(labels):
                fig.add_trace(
                    gobj.Scatter3d(
                        x=point_cloud[labels == label][:, 0],
                        y=point_cloud[labels == label][:, 1],
                        z=point_cloud[labels == label][:, 2],
                        mode="markers",
                        name=names[label],
                        marker={
                            "size": marker_size,
                            "symbol": symbol,
                            "color": label if indicate_labels else None,
                            "colorscale": "Viridis",
                            "opacity": opacity,
                        },
                    )
                )
        if to_scale:
            fig.update_layout(scene_aspectmode="data")

    # Title
    if title is not None:
        title_data = {
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
        fig.update_layout(title=title_data)

    # Radius
    if radius is not None:
        radius = np.array(radius)
        for ix, rad in enumerate(radius):
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=point_cloud[ix][0] - rad,
                y0=point_cloud[ix][1] - rad,
                x1=point_cloud[ix][0] + rad,
                y1=point_cloud[ix][1] + rad,
                **radius_kwargs,
            )

    # Lines
    if lines is not None:
        lines = np.array(lines)
        for line in lines:
            fig.add_trace(
                gobj.Scatter(
                    x=line[:, 0],
                    y=line[:, 1],
                    mode="lines",
                    line={"width": line_width, "color": "black"},
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    # Arrows
    if arrows is not None:
        annotations_arrows = []
        arrows = np.array(arrows)
        for arrow in arrows:
            arrow_data = gobj.layout.Annotation(
                dict(
                    x=arrow[1, 0],
                    y=arrow[1, 1],
                    xref="x",
                    yref="y",
                    showarrow=True,
                    axref="x",
                    ayref="y",
                    text="",
                    ax=arrow[0, 0],
                    ay=arrow[0, 1],
                    arrowhead=3,
                    arrowwidth=arrow_width,
                    arrowcolor="black",
                )
            )
            annotations_arrows.append(arrow_data)
        fig.update_layout(annotations=annotations_arrows)

    # Update trace and layout according to user input
    if colorscale == "wong":
        unique_labels = np.unique(labels)
        n = np.ptp(labels) + 1
        update = min(
            min([label.is_integer() for label in unique_labels]),
            n <= 8,
        )
        if not update:
            warnings.warn(
                '`colorscale` was set to `"wong"`, but the set of `labels` is '
                "not a set of at most eight consecutive integers; choice of "
                "`colorscale` is thus ignored."
            )
            pass
        else:
            n = int(n)
            d = cs_wong.dictionary
            ticks = np.repeat(np.linspace(0, 1, n + 1), 2)[1:-1].tolist()
            rgbs = np.repeat(
                np.array(
                    [
                        f"rgb{val}"
                        for val in list(d.values())[:n]
                    ],
                    dtype=object
                ),
                2
            ).tolist()
            fig.update_traces(marker_colorscale=list(zip(ticks, rgbs)))
    elif colorscale:
        fig.update_traces(marker_colorscale=colorscale)
    if with_colorbar:
        fig.update_traces(marker_showscale=True)
    if plotly_params:
        fig.update_traces(plotly_params.get("trace", None))
        fig.update_layout(plotly_params.get("layout", None))

    if display_plot:
        fig.show()
    return fig


def cloud_from_fcn_2_dim(
        fcn,
        min=0,
        max=1,
        steps=25,
):
    """Function generating a 2-dim. point cloud that allows plotting the graph
    of a real-valued function in x. Given a function f(x), where x is a real
    number, this function generates a NumPy-array of shape (steps, 2) where
    each entry is of the form [x, f(x)], and where `steps` denotes the number
    of points that are sampled from the range of x.

    Args:
        fcn (function): Real-valued function in one variable.
        min (float, optional): Minimum value for x. Defaults to 0.
        max (float, optional): Maximum value for x. Defaults to 1.
        steps (int): The number of steps into which the range for the x-values
            is to be subdivided. Must be a positive integer. Defaults to 25.

    Returns:
        numpy.ndarray of shape (n_points, 2): A NumPy-array of shape (steps, 2)
            where each entry is of the form [x, f(x)].
    """
    # Validate `fcn` parameter
    if not callable(fcn):
        raise ValueError(
            "The `fcn` parameter must be a callable."
            )
    # Validate `min` parameter
    if not (
        isinstance(min, (int, float))
        and min < float("inf")
        and min > -float("inf")
    ):
        raise ValueError(
            "The `min` parameter must be a finite integer or a float."
            )
    # Validate `max` parameter
    if not (
        isinstance(max, (int, float))
        and max < float("inf")
        and max > -float("inf")
    ):
        raise ValueError(
            "The `max` parameter must be a finite integer or a float."
            )
    # Validate `steps` parameter
    if not (
        isinstance(steps, int)
        and steps > 0
        and steps < float("inf")
    ):
        raise ValueError(
            "The `steps` parameter must be a finite positive integer."
            )

    data = np.array([
        [x, fcn(x)]
        for x in np.linspace(min, max, steps)
    ])
    return data


def cloud_from_fcn_3_dim(
        fcn,
        mins=[0, 0],
        maxs=[1, 1],
        steps=[25, 25],
):
    """Function generating a 3-dim. point cloud that allows plotting the graph
    of a real-valued function in x and y. Given a function f(x,y), where x and
    y are real numbers, this function generates a NumPy-array of shape
    (n_points, 3) where each entry is of the form [x, y, f(x,y)], and where
    n_points is determined by the range of x and y as well as the numbers of
    steps chosen. More precisely, n_points equals steps[0] * steps[1].

    Args:
        fcn (function): Real-valued function in two real variables.
        mins (list[float], optional): List or array containing the two minimum
            values for x and y, in this order. Defaults to [0, 0].
        maxs (list[float], optional): List or array containing the two maximum
            values for x and y, in this order. Defaults to [1, 1].
        steps (list[int], optional): List or array containing the number of
            steps into which the range for the x- and y-values are to be
            subdivided, in this order. Defaults to [25, 25].

    Returns:
        numpy.ndarray of shape (n_points, 3): A NumPy-array of shape
            (n_points, 3) where each entry is of the form [x, y, f(x,y)], and
            where n_points equals steps[0] * steps[1].
    """
    # Validate `fcn` parameter
    if not callable(fcn):
        raise ValueError(
            "The `fcn` parameter must be a callable."
            )
    # Validate `mins` parameter
    try:
        mins = np.asarray(mins, dtype=float)
    except Exception as exc:
        raise ValueError(
            "The `mins` parameter must be convertible into a "
            f"NumPy-array of floats, However, NumPy says: '{exc}'."
        )
    if not (
        mins.ndim == 1
        and len(mins) == 2
    ):
        raise ValueError(
            "The `min` parameter must contain precisely two values."
            )
    # Validate `maxs` parameter
    try:
        maxs = np.asarray(maxs, dtype=float)
    except Exception as exc:
        raise ValueError(
            "The `maxs` parameter must be convertible into a "
            f"NumPy-array of floats, However, NumPy says: '{exc}'."
        )
    if not (
        maxs.ndim == 1
        and len(maxs) == 2
    ):
        raise ValueError(
            "The `maxs` parameter must contain precisely two values."
            )
    # Validate `steps` parameter
    try:
        steps = np.asarray(steps, dtype=int)
    except Exception as exc:
        raise ValueError(
            "The `steps` parameter must be convertible into a "
            f"NumPy-array of ints, However, NumPy says: '{exc}'."
        )
    if not (
        steps.ndim == 1
        and len(steps) == 2
    ):
        raise ValueError(
            "The `steps` parameter must contain precisely two values."
            )
    data = np.array([
        np.array([
            [x, y, fcn(x, y)]
            for x in np.linspace(mins[0], maxs[0], steps[0])
        ])
        for y in np.linspace(mins[1], maxs[1], steps[1])
    ])
    return data.reshape(-1, 3)
