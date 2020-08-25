# -*- coding: utf-8 -*-
"""TRCA.

"""

import numpy as np

from scipy.linalg import eigh

from ..utils.base import nearestPD,crossdot
from .base import robustPattern, LinearFilter, linear_filter, FilterBank


def trca_kernel(Cs, Ct):
    """The kernel part in TRCA algorithm based on paper[1].
    
    Parameters
    ----------
    Cs : ndarray
        ERP template signal covariance, shape (n_classes, n_channels, n_channels)
    Ct : ndarray
        ERP template signal within-trial covariance, shape (n_classes, n_channels, n_channels)

    Returns
    -------
    Ws: ndarray
        spatial filters, shape (n_classes, n_filters, n_channels)
    Ds: ndarray
        eigenvalues of Ws, shape (n_classes, n_filters)
    As: ndarray
        spatial patterns, shape(n_classes, n_filters, n_channels), use robust method in paper[2].

    References
    ----------
    [1] Nakanishi, Masaki, et al. "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis." IEEE Transactions on Biomedical Engineering 65.1 (2018): 104-112.
    [2] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging." Neuroimage 87 (2014): 96-110.
    """
    Cs = np.reshape(Cs, (-1, *Cs.shape[-2:]))
    Ct = np.reshape(Ct, (-1, *Ct.shape[-2:]))

    Ws = []
    Ds = []
    As = []
    for S, Q in zip(Cs, Ct):
        # S, Q for original paper consistency
        # TODO: a little different from original paper(for better coding!), need verify, thouth i trust myself!
        S = S - Q
        Q = nearestPD(Q)
        D, W = eigh(S, Q)

        ix = np.argsort(D)[::-1]
        D = D[ix]
        W = W[:, ix].T
        A = robustPattern(W, S)

        Ws.append(W)
        Ds.append(D)
        As.append(A)

    Ws = np.stack(Ws)
    Ds = np.stack(Ds)
    As = np.stack(As)
    return Ws, Ds, As


class TRCA(LinearFilter):
    def __init__(self, n_filters=1):
        self.n_filters = n_filters

    def fit(self, X, y):
        Nt, Ne, Ns = X.shape
        self.classes_ = np.unique(y)

        X -= np.mean(X, axis=-1, keepdims=True)

        filters = []
        patterns = []
        evokeds = []

        Ps = np.zeros((len(self.classes_), Ne, Ns))
        Ct = np.zeros((len(self.classes_), Ne, Ne))
        for i, l in enumerate(self.classes_):
            Ps[i] = np.sum(X[y==l], axis=0)
            Ct[i] = np.sum(
                crossdot(X[y==l]),
                axis=0
            )

        Cs = crossdot(Ps)

        Ws, Ds, As = trca_kernel(Cs, Ct)

        Ws = Ws[:, :self.n_filters, :]
        Ds = Ds[:, :self.n_filters]
        As = As[:, :self.n_filters, :]

        evokeds = [linear_filter(P, W) for P, W in zip(Ps, Ws)]
        filters = np.reshape(Ws, (-1, Ws.shape[-1]))
        patterns = np.reshape(As, (-1, As.shape[-1]))
        self._set_filters(filters)
        self.patterns_ = patterns
        self.evokeds_ = np.concatenate(evokeds, axis=0)
        return self

    def transform(self, X):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X -= np.mean(X, axis=-1, keepdims=True)
        
        X_filt = super().transform(X)
        features = np.zeros((len(X_filt), len(self.filters_)))
        sigma2 = np.sqrt(
            np.sum(np.multiply(self.evokeds_, self.evokeds_), axis=-1)
        )
        for i, x in enumerate(X_filt):
            tmp = np.sum(np.multiply(x, self.evokeds_), axis=-1)
            sigma1 = np.sqrt(np.sum(np.multiply(x, x), axis=-1))
            tmp = tmp / sigma1 / sigma2
            features[i] = tmp
        return features


class FBTRCA(FilterBank):
    def __init__(self, coeffs, n_filters=1):
        self.coeffs = coeffs
        self.n_filters = n_filters
        super().__init__(
            TRCA(n_filters=n_filters)
        )

    def transform(self, X):
        features = super().transform(X)
        features = np.tensordot(features, self.coeffs, axes=(0, -1))
        return features
