import numpy as np
import pytest  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from dowker_complex import DowkerComplex  # type: ignore


@pytest.fixture
def random_data():
    n, dim = 100, 4
    ratio_vertices = 0.9
    X, y = (
        list(train_test_split(
            np.random.randn(n, dim), train_size=ratio_vertices)
        ),
        None,
    )
    return X, y


@pytest.fixture
def quadrilateral():
    vertices = np.array([
        [0, 0],
        [2, 0],
        [4, 2],
        [0, 4]
    ])
    witnesses = np.array([
        [2, 3],
        [0, 2],
        [1, 0],
        [3, 1]
    ])
    X, y = [vertices, witnesses], None
    return X, y


@pytest.fixture
def octagon():
    t = 1 / np.sqrt(2)
    vertices = np.array([
        [1, 0],
        [t, t],
        [0, 1],
        [-t, t]
    ])
    witnesses = np.array([
        [-1, 0],
        [-t, -t],
        [0, -1],
        [t, -t]
    ])
    X, y = [vertices, witnesses], None
    return X, y


def test_dowker_complex(random_data):
    """
    Check whether `DowkerComplex` runs at all for all admissible choices of
    `max_dimension` and produces complex of correct dimension.
    """
    X, y = random_data
    for max_dimension in [0, 1, 2]:
        drc = DowkerComplex(max_dimension=max_dimension)
        drc.fit_transform(X, y)
        assert hasattr(drc, "persistence_")
        assert hasattr(drc, "complex_")
        assert drc.complex_.dimension() == max_dimension + 1


def test_dowker_complex_cosine(random_data):
    """
    Check whether `DowkerComplex` runs on random data with non-default metric.
    """
    X, y = random_data
    drc = DowkerComplex(metric="cosine")
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")


def test_dowker_complex_empty_vertices():
    """
    Check whether `DowkerComplex` runs for empty set of vertices.
    """
    X, y = [np.random.randn(0, 512), np.random.randn(10, 512)], None
    drc = DowkerComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (
        drc.persistence_[0] == np.empty(
            (0, 2)
        )
    ).all()
    assert (
        drc.persistence_[1] == np.empty(
            (0, 2)
        )
    ).all()


def test_dowker_complex_empty_witnesses():
    """
    Check whether `DowkerComplex` runs for empty set of witnesses.
    """
    X, y = [np.random.randn(10, 512), np.random.randn(0, 512)], None
    drc = DowkerComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (
        drc.persistence_[0] == np.empty(
            (0, 2)
        )
    ).all()
    assert (
        drc.persistence_[1] == np.empty(
            (0, 2)
        )
    ).all()


def test_dowker_complex_plotting_2d(random_data):
    """
    Check whether `DowkerComplex` plots 2D data.
    """
    X, y = random_data
    X = [pt_cloud[:, :2] for pt_cloud in X]
    drc = DowkerComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    drc.plot_points()
    drc.plot_persistence()


def test_dowker_complex_plotting_3d(random_data):
    """
    Check whether `DowkerComplex` plots 3D data.
    """
    X, y = random_data
    X = [pt_cloud[:, :3] for pt_cloud in X]
    drc = DowkerComplex()
    drc.fit_transform(X, y)
    assert hasattr(drc, "persistence_")
    drc.plot_points()
    drc.plot_persistence()


def test_dowker_complex_quadrilateral(quadrilateral):
    """
    Check whether `DowkerComplex` returns correct result on small
    quadrilateral.
    """
    drc = DowkerComplex()
    drc.fit_transform(*quadrilateral)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    assert (
        drc.persistence_[0] == np.array(
            [[1, np.inf]],
        )
    ).all()
    assert (
        drc.persistence_[1] == np.array(
            [[np.sqrt(5), 3]],
        )
    ).all()


def test_dowker_complex_octagon(octagon):
    """
    Check whether `DowkerComplex` returns correct result on regular octagon.
    """
    drc = DowkerComplex()
    drc.fit_transform(*octagon)
    assert hasattr(drc, "persistence_")
    assert len(drc.persistence_) == 2
    birth = np.sqrt(2 - np.sqrt(2))
    death = np.sqrt(2 + np.sqrt(2))
    assert (
        drc.persistence_[0] == np.array([
            [birth, death],
            [birth, np.inf]
        ])
    ).all()
    assert (
        drc.persistence_[1] == np.empty(shape=(0, 2))
    ).all()
