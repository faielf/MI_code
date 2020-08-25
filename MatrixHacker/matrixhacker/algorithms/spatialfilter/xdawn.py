# -*- coding: utf-8 -*-
"""Xdawn.

"""

import numpy as np

from scipy.linalg import eigh

from ..utils.base import crossdot, nearestPD
from ..utils.covariance import covariances
from .base import linear_filter, LinearFilter, robustPattern


def xdawn_kernel(Cs, Cn):
    """The kernerl in Xdawn algorithm based on paper[1][2].
    
    Parameters
    ----------
    Cs : ndarray
        ERP template signal, shape (n_classes, n_channels, n_channels)
    Cn : ndarray
        noise signal, shape (n_channels, n_channels)
    
    Returns
    -------
    Ws: ndarray
        spatial filters, shape (n_classes, n_filters, n_channels)
    Ds: ndarray
        eigenvalues of Ws, shape (n_classes, n_filters)
    As: ndarray
        spatial patterns, shape(n_classes, n_filters, n_channels), use robust method in paper[3].

    References
    ----------
    [1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN algorithm to enhance evoked potentials: application to brain-computer interface. Biomedical Engineering, IEEE Transactions on, 56(8), 2035-2043.
    [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J. (2011, August). Theoretical analysis of xDAWN algorithm: application to an efficient sensor selection in a P300 BCI. In Signal Processing Conference, 2011 19th European (pp. 1382-1386). IEEE.
    [3] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging." Neuroimage 87 (2014): 96-110.
    """
    Cs = np.reshape(Cs, (-1, *Cs.shape[-2:]))
    Cn = nearestPD(Cn)

    Ws = []
    As = []
    Ds = []
    for C in Cs:
        D, W = eigh(C, Cn)
        ix = np.argsort(D)[::-1]
        D = D[ix]
        W = W[:, ix]
        W /= np.apply_along_axis(np.linalg.norm, 0, W)
        A = robustPattern(W, C)
        Ws.append(W.T)
        As.append(A.T)
        Ds.append(D.T)
    Ws = np.stack(Ws)
    As = np.stack(As)
    Ds = np.stack(Ds)
    return Ws, Ds, As


class Xdawn(LinearFilter):
    def __init__(self, n_filters=4, estimator='lwf', use_trace=False, baseline_cov=None):
        self.n_filters = n_filters
        self.estimator = estimator
        self.use_trace = use_trace
        self.baseline_cov = baseline_cov

    def fit(self, X, y):
        X = X.copy()
        y = y.copy()

        Nt, Ne, Ns = X.shape
        self.classes_ = np.unique(y)

        if self.baseline_cov is None:
            # TODO: is that reasonable?
            self.baseline_cov = covariances(np.mean(X, axis=0), estimator=self.estimator)

        if self.use_trace:
            self.baseline_cov /= np.trace(self.baseline_cov, axis1=-2, axis2=-1)

        filters = []
        patterns = []
        evokeds = []

        Ps = np.zeros((len(self.classes_), Ne, Ns))
        for i, l in enumerate(self.classes_):
            Ps[i] = np.mean(X[y==l], axis=0)

        Cs = covariances(Ps, estimator=self.estimator)

        if self.use_trace:
            Cs /= np.trace(Cs, axis1=-2, axis2=-1)[:, np.newaxis, np.newaxis]

        Ws, Ds, As = xdawn_kernel(Cs, self.baseline_cov)
        Ws = Ws[:, :self.n_filters, :]
        Ds = Ds[:, :self.n_filters]
        As = As[:, :self.n_filters, :]

        evokeds = [linear_filter(P, W) for P, W in zip(Ps, Ws)]
        filters = np.reshape(Ws, (-1, Ne))
        patterns = np.reshape(As, (-1, Ne))
        self._set_filters(filters)
        self.patterns_ = patterns
        self.evokeds_ = np.concatenate(evokeds, axis=0)
        return self

    def transform(self, X):
        X = X.copy()
        X = np.reshape(X, (-1, *X.shape[-2:]))
        features = super().transform(X)
        return features
