# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as linalg
from scipy.linalg import subspace_angles
from sklearn.svm import SVC

from .base import logm, sqrtm


def kl_divergence_mat(P, Q):
    """Kullback-Leibler divergence, asymmetric and nonnegative.
    Two distributions should be zero mean.
    
    Parameters
    ----------
    P : ndarray
        covariance matrix of distribution A, shape (N, N).
    Q : ndarray
        covariance matrix of distribution B, shape (N, N).
    
    Returns
    -------
    float
        KL-divergence.
    """
    dim = P.shape[0]
    logdet = np.log(linalg.det(Q) / linalg.det(P))
    kl = 0.5*(np.trace(linalg.inv(Q)@P) + logdet - dim)
    return kl


def sym_kl_divergence_mat(P, Q):
    """Symmetrized Kullback-Leibler divergence, symmetric and nonnegative.
    
    Parameters
    ----------
    P : ndarray
        covariance matrix of distribution A, shape (N, N).
    Q : ndarray
        covariance matrix of distribution B, shape (N, N).
    
    Returns
    -------
    float
        symmetric KL-divergence.
    """
    return kl_divergence_mat(P, Q) + kl_divergence_mat(Q, P)


def lambda_divergence_mat(P, Q, l=1.0):
    """Lambda divergence, symmetric and nonnegtive.
    
    Parameters
    ----------
    P : ndarray
        covariance matrix of distribution A, shape (N, N).
    Q : ndarray
        covariance matrix of distribution B, shape (N, N).
    l : float, optional
        lambda value, by default 1.0
    
    Returns
    -------
    float
        lambda-divergence.
    """
    M = l*P + (1-l)*Q
    return l*kl_divergence_mat(P, M) + (1-l)*kl_divergence_mat(Q, M)


def js_divergence_mat(P, Q):
    """Jensen-Shannon divergence, given by l=0.5 in Lambda divergence, symmetric and nonnegtive.
    
    Parameters
    ----------
    P : ndarray
        covariance matrix of distribution A, shape (N, N).
    Q : ndarray
        covariance matrix of distribution B, shape (N, N).

    Returns
    -------
    float
        Jensen-Shannon divergence.
    """
    return lambda_divergence_mat(P, Q, l=0.5)


def logeuclid_distance_mat(A, B):
    """Log Euclidean distance between matrices A and B.
    
    Parameters
    ----------
    A : ndarray
        SPD matrix A, shape (N, N).
    B : ndarray
        SPD matrix B, shape (N, N).
    
    Returns
    -------
    float
        Log Euclidean distance.
    """
    return linalg.norm(logm(A)-logm(B), ord='fro')


def wasserstein2_distance_mat(ma, A, mb, B):
    """2-Wasserstein distance between two distributions N(ma, A) and N(mb, B).
    
    Parameters
    ----------
    ma : ndarray
        mean vector of guassian distribution A.
    A : ndarray
        covariance matrix of guassian distribution A.
    mb : ndarray
        mean vector of guassian distribution B.
    B : ndarray
        covariance matrix of guassian distribution B.
    
    Returns
    -------
    float
        2-wasserstein distance.
    """
    B12 = sqrtm(B)
    C = sqrtm(B12@A@B12)
    w2 = linalg.norm(ma - mb, ord=2) + np.trace(A + B - 2*C)
    return np.sqrt(w2)


def principal_angle_mat(X, Y):
    """Principal angles between two subspaces X and Y.
    
    Parameters
    ----------
    X : ndarray
        feature vectors in M-dimensional space, shape (Nx, M).
    Y : ndarray
        feature vectors in M-dimensional space, shape (Ny, M).
    
    Returns
    -------
    float
        subspace angle in radian.
    """
    angles = subspace_angles(X.T, Y.T)
    return angles[0]


def proxy_a_distance_mat(Xs, Xt):
    """Proxy A-distance between two distributions Xs and Xt.
    Copy from code [1]_, using linear SVM as classifier.
    
    Parameters
    ----------
    Xs : ndarray
        source feature vectors, shape (Ns, M).
    Xt : ndarray
        target feature vectors, shape (Nt, M).
    
    Returns
    -------
    float
        A-distance between Xs and Xt.

    References
    ----------
    .. [1] https://github.com/jindongwang/transferlearning/blob/master/code/distance/proxy_a_distance.py
    """
    ns, nt = Xs.shape[0], Xt.shape[0]
    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(ns/2), int(nt/2)
    train_X = np.vstack((Xs[0:half_source, :], Xt[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((Xs[half_source:, :], Xt[half_target:, :]))
    test_Y = np.hstack((np.zeros(ns - half_source, dtype=int), np.ones(nt - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)
        
        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def _mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def _mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def _mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


_mmd_kernels = {
    "linear": _mmd_linear,
    "rbf": _mmd_rbf,
    "poly": _mmd_poly
}


def _check_kernel(mmd_kernel):
    """Check if a given estimator is valid.

    Parameters
    ----------
    mmd_kernel : callable object or str
        Could be the name of estimator or a callable estimator itself.

    Returns
    -------
    mmd_kernel: callable object
        A callable estimator.
    """
    if callable(mmd_kernel):
        pass
    elif mmd_kernel in _mmd_kernels.keys():
        mmd_kernel = _mmd_kernels[mmd_kernel]
    else:
        raise ValueError(
            """%s is not an valid kernel estimator ! Valid estimators are : %s or a
             callable function""" % (mmd_kernel, (' , ').join(_mmd_kernels.keys())))
    return mmd_kernel


def maximum_mean_discrepancy(X, Y, kernel='linear', **kwards):
    """Maximum mean discrepancy between distributions X and Y.
    
    Parameters
    ----------
    X : ndarray
        feature vectors, shape (Nx, M).
    Y : ndarray
        feature vectors, shape (Ny, M)
    kernel : str or callable object, optional
        used kernel, by default 'linear'.
    
    Returns
    -------
    float
        mmd distance.
    """
    mmd_kernel = _check_kernel(kernel)
    mmd = mmd_kernel(X, Y, **kwards)
    return mmd

