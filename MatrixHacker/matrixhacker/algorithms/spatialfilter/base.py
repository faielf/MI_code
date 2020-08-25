# -*- coding: utf-8 -*-
"""Basic methods for spatial filters.

"""
import numpy as np

from scipy.linalg import solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import clone


def linear_filter(X, filters):
    """Linear filter applied on data X.
    
    Parameters
    ----------
    X : ndarray
        Input data, shape (..., n_channels, n_samples).
    filters : ndarray
        Filters, shape (n_filters, n_channels).
    
    Returns
    -------
    Xf : ndarray
        Linear filtered data, shape (..., n_filters, n_samples).

    Notes
    -----
    .. math::
        \mathbf{Xf} = \mathbf{W} \mathbf{X}
    """
    Xf = np.swapaxes(
        np.tensordot(X, filters, axes=((-2), (-1))),
        -1, -2
    )
    return Xf


def bilinear_filter(X, filters):
    """Bilinear filter applied on data X.
    
    Parameters
    ----------
    X : ndarray
        Input covariance-like data, shape (..., n_channels, n_channels).
    filters : ndarray
        Filters, shape (n_filters, n_channels).
    
    Returns
    -------
    Xf : ndarray
        Bilinear filtered data, shape (..., n_filters, n_filters).

    Notes
    -----
    .. math::
        \mathbf{Xf} = \mathbf{W} \mathbf{X} \mathbf{W}^T
    """
    Xf = np.tensordot(
        linear_filter(X, filters), 
        filters, 
        axes=((-1), (-1))
    )
    return Xf


class LinearFilter(BaseEstimator, TransformerMixin):
    """Transform data with linear filters.
    
    Parameters
    ----------
    filters : ndarray
        Filters, shape (n_filters, n_channels).

    Notes
    -----
    Linear filtering looks like this:

    .. math::
        \mathbf{Xf} = \mathbf{W} \mathbf{X}
    """

    def __init__(self, filters):
        self._set_filters(filters)

    def fit(self, X, y=None):
        """Do nothing, be compatiable with sklearn API."""
        return self

    def transform(self, X):
        """Transform X with linear filters.
        
        Parameters
        ----------
        X : ndarray
            Input data, shape (...., n_channels, n_samples).
        
        Returns
        -------
        Xf : ndarray
            Linear filted data, shape (..., n_filters, n_samples).
        """
        X = X.copy()
        Xf = linear_filter(X, self.filters_)
        return Xf
    
    def _set_filters(self, filters):
        """Set filters"""
        self.filters_ = filters


class BilinearFilter(BaseEstimator, TransformerMixin):
    """Transform data with bilinear filters.
    
    Parameters
    ----------
    filters : ndarray
        Filters, shape (n_filters, n_channels).

    Notes
    -----
    Bilinear filtering looks like this:

    .. math::
        \mathbf{Xf} = \mathbf{W} \mathbf{X} \mathbf{W}^T 
    """

    def __init__(self, filters):
        self._set_filters(filters)

    def fit(self, X, y=None):
        """Do nothing, be compatiable with sklearn API."""
        return self

    def transform(self, X):
        """Transform X with bilinear filters.
        
        Parameters
        ----------
        X : ndarray
            Input covariance-like data, shape (...., n_channels, n_channels).
        
        Returns
        -------
        Xf : ndarray
            Bilinear filted data, shape (..., n_filters, n_filters).
        """
        X = X.copy()
        Xf = bilinear_filter(X, self.filters_)
        return Xf
    
    def _set_filters(self, filters):
        self.filters_ = filters


class FilterBank(BaseEstimator, TransformerMixin):
    """Apply a given indentical pipeline over a bank of filters.

    Parameters
    ----------
    estimator: sklean Estimator object
        The sklearn pipeline to apply on each band of the filter bank.

    Notes
    -----
    The pipeline provided with the constrictor will be appield on the 4th
    axis of the input data. This pipeline should be used with a FilterBank
    paradigm.

    This can be used to build a filterbank CSP, for example::

        pipeline = make_pipeline(FilterBank(estimator=CSP()), LDA())
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fit model with each band of X.
        
        Parameters
        ----------
        X : ndarray
            Filted data, shape (n_bands, ...) where the first dim must be the number of filters.
        y : None | ndarray, optional
            Labels of trials.
        
        Returns
        -------
        FilterBank object
            The FilterBank instance.
        """
        self.models = [
            clone(self.estimator).fit(X[i, ...], y)
            for i in range(len(X))
        ]
        return self

    def transform(self, X):
        """Transform each band of X with model.
        
        Parameters
        ----------
        X : ndarray
            Filted EEG data, shape (n_bands, ...) where the first dim must be the number of filters.
        
        Returns
        -------
        ndarray
            Transformed features stacked on the first dim, shape (n_bands, ...), the rest of shape is determinated by the model.
        """
        out = [self.models[i].transform(X[i, ...]) for i in range(len(X))]
        return np.stack(out, axis=0)


def robustPattern(W, C):
    """Transform spatial filters to spatial patterns based on paper [1]_.

    Parameters
    ----------
    W : ndarray
        Spatial filters, shape (n_filters, n_channels).
    C : ndarray
        Covariance matrix of A in generalize Rayleigh quotient, shape (n_channels, n_channels).

    Returns
    -------
    A : ndarray
        Spatial patterns, shape (n_filters, n_channels), each row is a spatial pattern.

    References
    ----------
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging." Neuroimage 87 (2014): 96-110.
    """
    Cs = np.dot(np.dot(W, C), W.T)

    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    # FIXME: Better than inv or pinv?
    A = solve(Cs.T, np.dot(C, W.T).T)
    return A    


def rjd(X, eps=1e-8, n_iter_max=1000):
    """Approximate joint diagonalization based on jacobi angle.

    Parameters
    ----------
    X : ndarray
        A set of covariance matrices to diagonalize, shape (n_trials, n_channels, n_channels).
    eps : float, optional
        Tolerance for stopping criterion (default 1e-8).
    n_iter_max : int, optional
        The maximum number of iteration to reach convergence (default 1000).

    Returns
    -------
    V : ndarray
        The diagonalizer, shape (n_filters, n_channels), usually n_filters == n_channels.
    D : ndarray
        The set of quasi diagonal matrices, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    This is a direct implementation of the Cardoso AJD algorithm [1]_ used in
    JADE. The code is a translation of the matlab code provided in the author
    website.

    References
    ----------
    .. [1] Cardoso, Jean-Francois, and Antoine Souloumiac. Jacobi angles for simultaneous diagonalization. SIAM journal on matrix analysis and applications 17.1 (1996): 161-164.

    """

    # reshape input matrix
    A = np.concatenate(X, 0).T

    # init variables
    m, nm = A.shape
    V = np.eye(m)
    encore = True
    k = 0

    while encore:
        encore = False
        k += 1
        if k > n_iter_max:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = np.arange(p, nm, m)
                Iq = np.arange(q, nm, m)

                # computation of Givens angle
                g = np.array([A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]])
                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton +
                                         np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                encore = encore | (np.abs(s) > eps)
                if (np.abs(s) > eps):
                    tmp = A[:, Ip].copy()
                    A[:, Ip] = c * A[:, Ip] + s * A[:, Iq]
                    A[:, Iq] = c * A[:, Iq] - s * tmp

                    tmp = A[p, :].copy()
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp

                    tmp = V[:, p].copy()
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp

    D = np.reshape(A, (m, int(nm / m), m)).transpose(1, 0, 2)
    return V, D


def ajd_pham(X, eps=1e-6, n_iter_max=100):
    """Approximate joint diagonalization based on pham's algorithm.

    Parameters
    ----------
    X : ndarray
        A set of covariance matrices to diagonalize, shape (n_trials, n_channels, n_channels).
    eps : float, optional 
        Tolerance for stoping criterion (default 1e-6).
    n_iter_max : int, optional
        The maximum number of iteration to reach convergence (default 1000).

    Returns
    -------
    V : ndarray
        The diagonalizer, shape (n_filters, n_channels), usually n_filters == n_channels.
    D : ndarray
        The set of quasi diagonal matrices, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    This is a direct implementation of the PHAM's AJD algorithm [1]_.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive definite Hermitian matrices." SIAM Journal on Matrix Analysis and Applications 22, no. 4 (2001): 1136-1152.

    """
     # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(n_iter_max):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D


def uwedge(X, init=None, eps=1e-7, n_iter_max=100):
    """Approximate joint diagonalization algorithm UWEDGE.

    Parameters
    ----------
    X : ndarray
        A set of covariance matrices to diagonalize, shape (n_trials, n_channels, n_channels).
    init : None | ndarray, optional
        Initialization for the diagonalizer, shape (n_channels, n_channels).
    eps : float, optional
        Tolerance for stoping criterion (default 1e-7).
    n_iter_max : int
        The maximum number of iteration to reach convergence (default 1000).

    Returns
    -------
    W_est : ndarray
        The diagonalizer, shape (n_filters, n_channels), usually n_filters == n_channels.
    D : ndarray
        The set of quasi diagonal matrices, shape (n_trials, n_channels, n_channels).

    Notes
    -----
    Uniformly Weighted Exhaustive Diagonalization using Gauss iteration
    (U-WEDGE). Implementation of the AJD algorithm by Tichavsky and Yeredor [1]_ [2]_.
    This is a translation from the matlab code provided by the authors.

    References
    ----------
    .. [1] P. Tichavsky, A. Yeredor and J. Nielsen, "A Fast Approximate Joint Diagonalization Algorithm Using a Criterion with a Block Diagonal Weight Matrix", ICASSP 2008, Las Vegas.
    .. [2] P. Tichavsky and A. Yeredor, "Fast Approximate Joint Diagonalization Incorporating Weight Matrices" IEEE Transactions of Signal Processing, 2009.
    
    """
    L, d, _ = X.shape

    # reshape input matrix
    M = np.concatenate(X, 0).T

    # init variables
    d, Md = M.shape
    iteration = 0
    improve = 10

    if init is None:
        E, H = np.linalg.eig(M[:, 0:d])
        W_est = np.dot(np.diag(1. / np.sqrt(np.abs(E))), H.T)
    else:
        W_est = init

    Ms = np.array(M)
    Rs = np.zeros((d, L))

    for k in range(L):
        ini = k*d
        Il = np.arange(ini, ini + d)
        M[:, Il] = 0.5*(M[:, Il] + M[:, Il].T)
        Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
        Rs[:, k] = np.diag(Ms[:, Il])

    crit = np.sum(Ms**2) - np.sum(Rs**2)
    while (improve > eps) & (iteration < n_iter_max):
        B = np.dot(Rs, Rs.T)
        C1 = np.zeros((d, d))
        for i in range(d):
            C1[:, i] = np.sum(Ms[:, i:Md:d]*Rs, axis=1)

        D0 = B*B.T - np.outer(np.diag(B), np.diag(B))
        A0 = (C1 * B - np.dot(np.diag(np.diag(B)), C1.T)) / (D0 + np.eye(d))
        A0 += np.eye(d)
        W_est = np.linalg.solve(A0, W_est)

        Raux = np.dot(np.dot(W_est, M[:, 0:d]), W_est.T)
        aux = 1./np.sqrt(np.abs(np.diag(Raux)))
        W_est = np.dot(np.diag(aux), W_est)

        for k in range(L):
            ini = k*d
            Il = np.arange(ini, ini + d)
            Ms[:, Il] = np.dot(np.dot(W_est, M[:, Il]), W_est.T)
            Rs[:, k] = np.diag(Ms[:, Il])

        crit_new = np.sum(Ms**2) - np.sum(Rs**2)
        improve = np.abs(crit_new - crit)
        crit = crit_new
        iteration += 1

    D = np.reshape(Ms, (d, L, d)).transpose(1, 0, 2)
    return W_est, D


ajd_methods = {
    'rjd': rjd, 
    'ajd_pham': ajd_pham, 
    'uwedge': uwedge
}


def _check_ajd_method(method):
    """Check if a given method is valid.

    Parameters
    ----------
    method : callable object or str
        Could be the name of ajd_method or a callable method itself.

    Returns
    -------
    method: callable object
        A callable ajd method.
    """
    if callable(method):
        pass
    elif method in ajd_methods.keys():
        method = ajd_methods[method]
    else:
        raise ValueError(
            """%s is not an valid method ! Valid methods are : %s or a
             callable function""" % (method, (' , ').join(ajd_methods.keys())))
    return method


def ajd(X, method='uwedge'):
    """Wrapper of AJD methods.
    
    Parameters
    ----------
    X : ndarray
        Input covariance matrices, shape (n_trials, n_channels, n_channels)
    method : str, optional
        AJD method (default uwedge).
    
    Returns
    -------
    V : ndarray
        The diagonalizer, shape (n_filters, n_channels), usually n_filters == n_channels.
    D : ndarray
        The set of quasi diagonal matrices, shape (n_trials, n_channels, n_channels).
    """
    method = _check_ajd_method(method)
    V, D = method(X)
    return V, D

