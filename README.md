An implementation of the Dowker complex originally introduced in [<em>Homology Groups of Relations</em>](https://www.jstor.org/stable/1969768) and adapted to the setting of persistent homology in [<em>A functorial Dowker theorem and persistent homology of asymmetric networks</em>](https://link.springer.com/article/10.1007/s41468-018-0020-6).
The complex is implemented as a class named `DowkerComplex` that largely follows the API conventions from `scikit-learn`.

---

__Example of running DowkerComplex__

```
>>> from dowker_complex import DowkerComplex
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(
        n_samples=200,
        centers=[[-1, 0], [1, 0]],
        cluster_std=0.75,
        random_state=42,
    )
>>> vertices, witnesses = X[y == 0], X[y == 1]
>>> drc = DowkerComplex()  # use default parameters
>>> persistence = drc.fit_transform([vertices, witnesses])
>>> persistence
    [array([[0.39632083, 0.4189592 ],
            [0.17218397, 0.24239225],
            [0.07438909, 0.1733489 ],
            [0.13146844, 0.25247844],
            [0.16269607, 0.29266369],
            [0.0815455 , 0.24042536],
            [0.10576964, 0.32222553],
            [0.1382231 , 0.358332  ],
            [0.07358198, 0.37408252],
            [0.24082383, 0.57726198],
            [0.02419385,        inf]]),
    array([[0.5035793 , 0.63405836]])]
```

Any `DowkerComplex` object accepts further parameters during instantiation.
A full description of these can be displayed by calling `help(DowkerComplex)`.
These parameters, among other things, allow the user to specify persistence-related parameters such as the maximal homological dimension to compute or which metric to use.

---

__Requirements__

Required Python dependencies are specified in `pyproject.toml`.
Provided that `uv` is installed, these dependencies can be installed by running `uv pip install -r pyproject.toml`.
The environment specified in `uv.lock` can be recreated by running `uv sync`.

---

__Example of installing `dowker-complex` from source for `uv` users__

```
$ git clone github.com/m-a-huber/dowker-complex
$ cd dowker-complex
$ uv sync --no-dev
$ source .venv/bin/activate
$ python
>>> from dowker-complex import DowkerComplex
>>> ...
```
