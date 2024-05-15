import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from typing import Optional

_available_methods = ["hager", "cg", "spg"]


def _unit_vector(n: int, i: int) -> NDArray:
    """Creates a unit vector of size n."""
    x = np.zeros(n)
    x[i] = 1
    return x


def _hagers_estimate(
    LU: sparse.linalg.SuperLU, maxit: Optional[int] = None
) -> float:
    r"""
    Estimates $||A^{-1}||_1$ using Hager's method [H84] and the
    refinements proposed by Higham [H88].

    Parameters
    ----------
    LU : sparse.linalg.SuperLU
        LU decomposition of A.
    maxit : int, optional
        Maximum number of iterations. The default value is 5, based on [2].

    Returns
    -------
    gamma : float
        Hager's estimate for $||A^{-1}||_1$.


    References
    ----------
    [H84]:   Hager, W. W. (1984). Condition estimates. SIAM Journal on
             scientific and statistical computing, 5(2), 311-316.

    [H88]:   Higham, N. J. (1988). FORTRAN codes for estimating the one-norm of
             a real or complex matrix, with applications to condition
             estimation. ACM Transactions on Mathematical Software (TOMS),
             14(4), 381-396.

    [BHJ12]: Brás, C. P., Hager, W. W., & Júdice, J. J. (2012). An
             investigation of feasible descent algorithms for estimating the
             condition number of a matrix. Top, 20(3), 791-809.
    """
    if maxit is None:
        maxit = 5
    n = LU.shape[0]
    x = np.ones(n) / n

    gamma = 0
    xi = np.zeros(n)
    for i in range(maxit):
        oldxi = xi
        oldgamma = gamma

        y = LU.solve(x)

        gamma = np.linalg.norm(y, ord=1)
        xi = np.sign(y)
        if np.all(xi == oldxi) or gamma <= oldgamma:
            break

        z = LU.solve(xi, "T")
        absz = np.abs(z)
        maxidx = np.argmax(absz)
        if absz[maxidx] <= np.dot(x, z):
            break
        else:
            x = _unit_vector(n, maxidx)

    # this alternative x tests if the matrix is sensitive to cancellation
    ones = np.ones(n)
    arange = np.arange(n)
    xnew = (-ones) ** (arange) * (ones + arange / (n - 1))

    y = LU.solve(xnew)
    othergamma = 2 * np.linalg.norm(y, ord=1) / (3 * n)
    if othergamma > gamma:
        gamma = othergamma
    return gamma


class NonConvergenceError(Exception):
    pass


def _project_to_simplex(u: NDArray, p: float = 1) -> NDArray:
    r"""
    Block pivotal principal pivoting algorithm to project a vector $u$ onto the
    simplex $\{e^T x = p\}$ from [JRRS08].

    Parameters
    ----------
    u : np.array
        Vector to project.
    p : float, optional
        Size of the simplex, default is 1 (standard simplex).

    Returns
    -------
    z : np.array
       Projection of u onto the simplex.

    References
    ----------

    [JRRS08]: Júdice, J. J., Raydan, M., Rosa, S. S., & Santos, S. A. (2008).
              On the solution of the symmetric eigenvalue complementarity
              problem by the spectral projected gradient algorithm. Numerical
              Algorithms, 47, 391-407.
    """
    n = len(u)
    q = -u

    # step 0: instead of using a set of indices, we use an indicator array
    F_ind = np.ones(n, dtype=int)
    for i in range(n):

        # step 1
        phi = -(p + np.sum(q * F_ind)) / np.sum(F_ind)

        # step 2
        z = (q + phi) * F_ind
        if np.all(z == 0):
            # Due to rounding issues it can happen that z is all zero here.
            # Specifically, this happens if there is only one non-zero entry
            # left in F_ind and p + sum(q * F_ind) == sum(q * F_ind).
            # In this case, the proper solution is to return F_ind, because
            # this is the edge of the simplex that is still remaining.
            return F_ind

        H_ind = F_ind * (z > 0)
        if not np.any(H_ind):
            return -z
        else:
            F_ind[H_ind == 1] = 0
    raise NonConvergenceError("No convergence in projection algorithm.")


def _spectral_projected_gradient_estimate(
    LU: sparse.linalg.SuperLU, maxit: Optional[int] = None
) -> float:
    r"""
    Spectral projected gradient algorithm to estimate the 1-norm of
    $||A^{-1}||$ as described in [BHJ12].

    Parameters
    ----------
    LU : scipy.sparse.linalg.SuperLU
       LU decomposition of matrix A.
    maxit : int, optional
       Maximum number of iterations. Set to 100 by default.

    Returns
    -------
    gamma : float
        Estimate for $||A^{-1}||_1$.

    References
    ----------
    [BHJ12]: Brás, C. P., Hager, W. W., & Júdice, J. J. (2012). An
             investigation of feasible descent algorithms for estimating the
             condition number of a matrix. Top, 20(3), 791-809.
    """

    etamin = 1e-16
    etamax = 1e16
    eps = 1e-10

    if maxit is None:
        maxit = 100

    n = LU.shape[0]
    x = np.ones(n) / n
    z = x
    y = x
    xold = x
    for k in range(maxit):
        y = LU.solve(x)
        xi = np.sign(y)

        zold = z
        z = LU.solve(xi, "T")

        zmax = np.max(z)
        if zmax <= np.dot(z, x):
            return np.linalg.norm(y, ord=1)
        else:
            # find new search direction
            if k == 0:
                u = x + z
                pu = _project_to_simplex(u)
                eta = np.clip(
                    1 / np.linalg.norm(pu - x, ord=np.inf), etamin, etamax
                )
            else:
                s = x - xold
                yold = zold - z
                sy = np.dot(s, yold)
                ss = np.dot(s, s)
                if sy > eps:
                    eta = np.clip(ss / sy, etamin, etamax)
                else:
                    eta = etamax
            w = x + eta * z
            pw = _project_to_simplex(w)
            xold = x
            x = pw
    raise NonConvergenceError(
        "No convergence in spectral projected gradient algorithm."
    )


def _conditional_gradient_estimate(
    LU: sparse.linalg.SuperLU, maxit: Optional[int] = None
) -> float:
    r"""
    Conditional gradient algorithm to estimate the 1-norm of $||A^{-1}||$
    as described in [BHJ12].

    Parameters
    ----------
    LU : scipy.sparse.linalg.SuperLU
       LU decomposition of matrix A.
    maxit : int, optional
       Maximum number of iterations. Set to 5 by default.

    Returns
    -------
    gamma : float
        Estimate for $||A^{-1}||_1$.

    References
    ----------
    [BHJ12]: Brás, C. P., Hager, W. W., & Júdice, J. J. (2012). An
             investigation of feasible descent algorithms for estimating the
             condition number of a matrix. Top, 20(3), 791-809.
    """
    if maxit is None:
        maxit = 5
    n = LU.shape[0]
    x = np.ones(n) / n

    xi = np.zeros(n)
    for i in range(maxit):

        y = LU.solve(x)
        xi = np.sign(y)
        z = LU.solve(xi, "T")

        maxidx = np.argmax(z)
        if z[maxidx] <= np.dot(z, x):
            gamma = np.linalg.norm(y, ord=1)
            return gamma
        else:
            x = _unit_vector(n, maxidx)
    raise NonConvergenceError(
        "No convergence in conditional gradient algorithm."
    )


def cond1est(
    A: sparse.spmatrix, method: str = "hager", maxit: Optional[int] = None
) -> float:
    r"""
    Estimates the order-1 condition number of a sparse matrix:

    $$\kappa(A) = ||A||_1 \cdot ||A^{-1}||_1$$

    based on an approximation of ||A^{-1}||_1.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The matrix for which the condition number should be estimated.
        Preferably in CSC format, otherwise the matrix will be converted
        internally.
    method : str, optional (default: "hager")
        Which method to use to estimate $||A^{-1}||$:

        * "hager": Hager's method [H84], with refinements as proposed
          by Higham [H88]. This is the same method as used in Matlab and
          Octave's ``condest`` function.
        * "spg": Spectral projected gradient algorithm (see [BHJ12] for
          a detailed description and further references). Experimental.
        * "cg": Conditional gradient algorithm (see [BHJ12] for a detailed
          description and further references). Experimental.

    maxit : int, optional (default: None)
        The maximum number of iterations to use. The default value depens on
        the chosen method:

        * "hager": 5 (as proposed in [H88])
        * "cg": 5
        * "spg": 100

    Returns
    -------
    cond : float
        An estimate of the order-1 condition number.

    References
    ----------
    [H84]:   Hager, W. W. (1984). Condition estimates. SIAM Journal on
             scientific and statistical computing, 5(2), 311-316.

    [H88]:   Higham, N. J. (1988). FORTRAN codes for estimating the one-norm of
             a real or complex matrix, with applications to condition
             estimation. ACM Transactions on Mathematical Software (TOMS),
             14(4), 381-396.

    [BHJ12]: Brás, C. P., Hager, W. W., & Júdice, J. J. (2012). An
             investigation of feasible descent algorithms for estimating the
             condition number of a matrix. Top, 20(3), 791-809.
    """

    n = A.shape[0]
    assert n == A.shape[1], "A must be symmetric"
    normA = sparse.linalg.norm(A, ord=1)
    LU = sparse.linalg.splu(A)

    if method == "hager":
        gamma = _hagers_estimate(LU, maxit=maxit)
    elif method == "cg":
        gamma = _conditional_gradient_estimate(LU, maxit=maxit)
    elif method == "spg":
        gamma = _spectral_projected_gradient_estimate(LU, maxit=maxit)
    else:
        raise ValueError(f"Unknown method: {method}")
    return gamma * normA
