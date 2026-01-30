An implementation of the Dowker complex originally introduced in [<em>Homology Groups of Relations</em>](https://www.jstor.org/stable/1969768) and adapted to the setting of persistent homology in [<em>A functorial Dowker theorem and persistent homology of asymmetric networks</em>](https://link.springer.com/article/10.1007/s41468-018-0020-6).
The complex is implemented as a class named `DowkerComplex` that largely follows the API conventions from `scikit-learn`.

---

__Example of running DowkerComplex__

The following is an example of computing persistent homology of the filtered complex $`\left\{\mathrm{D}_{\varepsilon}(X,Y)\right\}_{\varepsilon\in\mathbb{R}^{+}}`$, that is, of the Dowker complex with relations $`R_{\varepsilon}\subseteq X\times Y`$ defined by $`(x,y)\in R_{\varepsilon}`$ iff $`d(x,y)\leq\varepsilon`$ for $`\varepsilon\geq 0`$, and where $`X`$ and $`Y`$ are subsets of $`\mathbb{R}^{n}`$ equipped with the Euclidean norm.
In the following example, we refer to $`X`$ and $`Y`$ as vertices and witnesses, respectively.

```
>>> from dc import DowkerComplex
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(
        n_samples=200,
        centers=[[-1, 0], [1, 0]],
        cluster_std=0.75,
        random_state=42,
    )
>>> vertices, witnesses = X[y == 0], X[y == 1]
>>> dc = DowkerComplex()  # use default parameters
>>> persistence = dc.fit_transform([vertices, witnesses])
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

The output above is a list of arrays, where the $`i`$-th array contains (birth, death)-times of homological generators in dimension $`i-1`$.
Validity of Dowker duality can be verified by swapping the roles of vertices as witnesses as follows.

```
>>> import numpy as np
>>> persistence_swapped = DowkerComplex().fit_transform([witnesses, vertices])
>>> all(
        np.allclose(homology, homology_swapped)
        for homology, homology_swapped
        in zip(persistence, persistence_swapped)
    )
True
```

Any `DowkerComplex` object accepts further parameters during instantiation.
A full description of these can be displayed by calling `help(DowkerComplex)`.
These parameters, among other things, allow the user to specify persistence-related parameters such as the maximal homological dimension to compute or which metric to use.

---

__Installation and requirements__

The package can be installed via `pip` by running `pip install -U dowker-complex`.

Required Python dependencies are specified in `pyproject.toml`.
Provided that `uv` is installed, these dependencies can be installed by running `uv pip install -r pyproject.toml`.
The environment specified in `uv.lock` can be recreated by running `uv sync`.

---

__Installing from PyPI for `uv` users__

```
$ uv init
$ uv add dowker-complex
$ uv run python
>>> from dc import DowkerComplex
>>> ...
```
