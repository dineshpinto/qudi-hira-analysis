import numpy as np
from scipy import sparse


def decibelm_to_watts(dbm_value: float) -> float:
    """ Convert dBm to Watts.  1 W = 10^((1 dBm - 30) / 10) """
    return 10 ** ((dbm_value - 30) / 10)


def format_exponent_as_str(number_to_format: float, decimals: int = 2) -> str:
    """
    Format an exponent as a LaTeX string
    e.g. 0.0001 will be formatted as $1.0 \times 10^{-4}$
    """
    count = 0

    if number_to_format > 1:
        while number_to_format >= 10:
            number_to_format /= 10
            count += 1
    else:
        while number_to_format < 1:
            number_to_format *= 10
            count -= 1

    formatted_str = r"${{ {} }} \times 10^{{ {} }}$".format(round(number_to_format, decimals), count)
    return formatted_str


def baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.9, niter: int = 10) -> np.ndarray:
    """
    Asymmetric least squares baseline.
    Source: Paul H. C. Eilers, Hans F.M. Boelens. Baseline Correction with Asymmetric Least Squares Smoothing (2005).
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)

    z = None
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z
