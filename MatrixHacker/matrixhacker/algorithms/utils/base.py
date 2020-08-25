"""Basic matrix operation methods.
"""
import warnings
import numpy as np
from numpy.core.numerictypes import typecodes
from scipy.linalg import eigh, eigvalsh


def crossdot(X):
    """Wrapper for high dimensional :math:`\mathbf{X} \mathbf{X}^T`.
    
    Parameters
    ----------
    X : ndarray
        High dimensional EEG matrix, shape (..., n_channels, n_samples).
    
    Returns
    -------
    cov_X : ndarray
        Matrix multiplication result of X, shape (..., n_channels, n_channels).
    """
    shape = X.shape
    X = np.reshape(X, (-1, *shape[-2:]))
    cov_X = np.stack([x.dot(x.T) for x in X])
    cov_X = np.reshape(cov_X, (*shape[:-2], *cov_X.shape[-2:]))
    return cov_X


def isPD(B):
    """Returns true when input matrix is positive-definite, via Cholesky decompositon method.
    
    Parameters
    ----------
    B : ndarray
        Any matrix, shape (N, N)
    
    Returns
    -------
    bool
        True if B is positve-definite.

    Notes
    -----
        Use numpy.linalg rather than scipy.linalg. In this case, scipy.linalg has unpredictable behaviors.
    """

    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(A):
    """Find the nearest positive-definite matrix to input.

    Parameters
    ----------
    A : ndarray
        Any square matrxi, shape (N, N)
    
    Returns
    -------
    A3 : ndarray
        positive-definite matrix to A

    Notes
    -----
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1]_, which
    origins at [2]_.

    References
    ----------
    .. [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    .. [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """     

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    print("Replace current matrix with the nearest positive-definite matrix.")

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `numpy.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def matrix_operator(Ci, operator):
    """Apply operator to any matrix.
    
    Parameters
    ----------
    Ci : ndarray
        Input positive definite matrix.
    operator : callable object
        Operator function or callable object.
    
    Returns
    -------
    Co : ndarray
        Operated matrix.
    
    Raises
    ------
    ValueError
        If Ci is not positive definite.

    Notes
    -----
    .. math::
        \mathbf{Ci} = \mathbf{V} \left( \mathbf{\Lambda} \\right) \mathbf{V}^T \\\\
        \mathbf{Co} = \mathbf{V} operator\left( \mathbf{\Lambda} \\right) \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    # FIXME: should Ci be positive-definite
    # if not isPD(Ci):
    #     raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")

    eigvals, eigvects = eigh(Ci, check_finite=False)
    eigvals = np.diag(operator(eigvals))
    Co = np.dot(np.dot(eigvects, eigvals), eigvects.T)
    return Co


def sqrtm(Ci):
    """Return the matrix square root of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Square root matrix of Ci.

    Notes
    -----
    .. math:: 
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return matrix_operator(Ci, np.sqrt)


def logm(Ci):
    """Return the matrix logrithm of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Logrithm matrix of Ci.

    Notes
    -----
    .. math:: 
        \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return matrix_operator(Ci, np.log)


def expm(Ci):
    """Return the matrix exponential of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Exponential matrix of Ci.

    Notes
    -----
    .. math:: 
        \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    return matrix_operator(Ci, np.exp)


def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.

    Returns
    -------
    ndarray
        Inverse matrix square root of Ci.

    Notes
    -----
    .. math:: 
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    isqrt = lambda x: 1. / np.sqrt(x)
    return matrix_operator(Ci, isqrt)


def powm(Ci, alpha):
    """Return the matrix power of a covariance matrix.

    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrix.
    alpha : float
        Exponent.

    Returns
    -------
    ndarray
        Power matrix of Ci.

    Notes
    -----
    .. math:: 
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`.
    """
    power = lambda x: x**alpha
    return matrix_operator(Ci, power)


def whitenm(Ci, R):
    """Whitening matrix Ci with reference matrix R.
    
    Parameters
    ----------
    Ci : ndarray
        Input positive-definite matrices, which references to R, shape (n_trials, n_channels, n_channels).
    R : ndarray
        The reference positive-definite matrix R, shape (n_channels, n_channels).
    Returns
    -------
    wCi : ndarray
        Whitening matrix of Ci, which references to Identity marix, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    The idea origins in cross-subject transfer learning [1]_, which also performs well at cross-session situation [2]_.
    The key is to map current covariance matrix with respect to an uniform baseline matrix (Identity matrix) 
    so that we can compare them under the same coordinate system.
    Thanks to the congruence invariance property of the Riemannian distance, 
    the distances between points of the same session or subject remains unchanged.

    .. math:: 
        \mathbf{wCi} = \mathbf{R}^{-1/2} \left( \mathbf{Ci} \\right) \mathbf{R}^{-1/2}

    References
    ----------
    .. [1] Reuderink, Boris, et al. "A subject-independent brain-computer interface based on smoothed, second-order baselining." 2011 Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE, 2011.
    .. [2] Zanini, Paolo, et al. "Transfer learning: a Riemannian geometry framework with applications to brain–computer interfaces." IEEE Transactions on Biomedical Engineering 65.5 (2017): 1107-1116.
    """
    n_trials, n_channels, _ = Ci.shape 
    R12 = invsqrtm(R)
    wCi = np.zeros((n_trials, n_channels, n_channels))
    for i, C in enumerate(Ci):
        wCi[i] = R12*C*R12
    return wCi


def unwhitenm(wCi, R):
    """Unwhitening matrix Ci with reference matrix R.
    
    Parameters
    ----------
    wCi : ndarray
        Input positive-definite matrices, which references to Identity matrix (already whitened), shape (n_trials, n_channels, n_channels).
    R : ndarray
        The reference positive-definite matrix R, shape (n_channels, n_channels).

    Returns
    -------
    Ci : ndarray
        Unwhitening matrix of Ci, which references to R, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    .. math:: 
        \mathbf{Ci} = \mathbf{R}^{1/2} \left( \mathbf{wCi} \\right) \mathbf{R}^{1/2}
    """
    n_trials, n_channels, _ = wCi.shape 
    R12 = sqrtm(R)
    Ci = np.zeros((n_trials, n_channels, n_channels))
    for i, C in enumerate(wCi):
        Ci[i] = R12*C*R12
    return Ci


def sign_flip(u, s, vh=None):
    """Flip signs of SVD or EIG using the method in paper [1]_.

    Parameters
    ----------
    u: ndarray
        left singular vectors, shape (M, K).
    s: ndarray
        singular values, shape (K,).
    vh: ndarray or None
        transpose of right singular vectors, shape (K, N).

    Returns
    -------
    u: ndarray
        corrected left singular vectors.
    s: ndarray
        singular values.
    vh: ndarray
        transpose of corrected right singular vectors.

    References
    ----------
    .. [1] https://www.sandia.gov/~tgkolda/pubs/pubfiles/SAND2007-6422.pdf
    """
    if vh is None:
        vh = u.T

    left_proj = np.sum(s[:, np.newaxis]*vh, axis=-1)
    right_proj = np.sum(u*s, axis=0)
    total_proj = left_proj + right_proj
    signs = np.sign(total_proj)
    
    random_idx = (signs==0)
    if np.any(random_idx):
        signs[random_idx] = 1
        warnings.warn("The magnitude is close to zero, the sign will become arbitrary.")

    u = u*signs
    vh = signs[:, np.newaxis]*vh

    return u, s, vh