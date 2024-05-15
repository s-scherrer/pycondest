import numpy as np
from pathlib import Path
import pytest
from scipy import sparse, io


testdatapath = Path(__file__).parent / "data"


test_matrices = {
    "random1": ("random", (10, 0.5, 1)),
    "random2": ("random", (10, 0.5, 1e-1)),
    "random3": ("random", (10, 0.5, 1e-2)),
    "random4": ("random", (100, 1e-2, 1e-1)),
    "random5": ("random", (100, 1e-2, 1e-2)),
    "random6": ("random", (100, 1e-3, 1e-2)),
    "random7": ("random", (1000, 1e-4, 1e-1)),
    "random8": ("random", (1000, 1e-4, 1e-2)),
    "random9": ("random", (1000, 1e-5, 1e-3)),
    "random10": ("random", (10000, 1e-5, 1e-3)),
    "random11": ("random", (10000, 1e-5, 1e-4)),
    "random12": ("random", (10000, 1e-5, 1e-5)),
    "random13": ("random", (20000, 1e-7, 1e-4)),
    "random14": ("random", (20000, 1e-7, 1e-5)),
}


def create_test_matrix_data(kind, params):
    print(f"Creating test matrix of type '{kind}' with parameters: {params}")
    # create test matrix
    if kind == "random":
        A = create_random_test_matrix(params)
    else:
        raise ValueError(f"No such test matrix type: {kind}")

    # get reference results from dense linalg and/or from octave
    data = {"A": A}
    if A.shape[0] <= 10000:
        # for larger matrices this can be too expensive
        data["cond1_dense"] = np.linalg.cond(A.toarray(), p=1)
    from oct2py import octave

    data["cond1_octave"] = octave.feval("condest", A)
    return data


def create_random_test_matrix(params):
    """
    Creates a random matrix with additional values on the diagonal
    (otherwise the matrix is too ill-conditioned to solve)
    """
    n, density, diagval = params
    np.random.seed(42)
    A = (
        sparse.random(
            n,
            n,
            density=density,
            format="csc",
            random_state=np.random.default_rng(seed=42),
        )
        + sparse.eye(n) * diagval
    )
    return A


@pytest.fixture(scope="package", params=list(test_matrices))
def test_matrix_data(request):
    name = request.param
    testdatapath.mkdir(exist_ok=True, parents=True)
    fname = testdatapath / f"{name}.mat"
    if not fname.exists():
        data = create_test_matrix_data(*test_matrices[name])
        print(f"Storing test data in {fname}")
        io.savemat(fname, data)
    return io.loadmat(fname)
