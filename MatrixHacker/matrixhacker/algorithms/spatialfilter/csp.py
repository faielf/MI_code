# -*- coding: utf-8 -*-
"""Common Spatial Pattern and  his happy little buddies!

"""

import numpy as np

from scipy.linalg import eigh, pinv
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from ..utils.base import nearestPD
from .base import robustPattern, BilinearFilter, FilterBank, ajd


def csp_kernel(C1, C2):
    """The kernel in CSP algorithm based on paper [1]_.

    Parameters
    ----------
    C1: ndarray
        Covariance of EEG training data1, shape (n_channels, n_channels).
    C2: ndarray
        Covariance of EEG training data2, shape (n_channels, n_channels).

    Returns
    -------
    W: ndarray
        Spatial filters, shape (n_filters, n_channels).
    D: ndarray
        Eigenvalues of W, shape (n_filters,).
    A: ndarray
        Spatial patterns, shape (n_filters, n_channels), use robust method in paper [2]_.

    References
    ----------
    .. [1] Parra, Lucas C., et al. "Recipes for the linear analysis of EEG." Neuroimage 28.2 (2005): 326-341.
    .. [2] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging." Neuroimage 87 (2014): 96-110.
    """
    Cc = C1 + C2
    Cc = nearestPD(Cc)

    # generalized eigenvalue problem
    D, W = eigh(C1, Cc)
    W = W[:, D>0]
    D = D[D>0]
    W = W.T

    # FIXME: check effectiveness for this method
    A = robustPattern(W, C1)
    return W, D, A


class CSP(BilinearFilter):
    """2-class Common Spatial Pattern.
    
    Parameters
    ----------
    n_filters : int, optional
        The number of CSP filters to use (default 2).
    use_trace : bool, optional
        If True, use trace operation to normalize covariance matrices.
    
    """
    def __init__(self, n_filters=2, use_trace=True):
        self.n_filters = n_filters
        self.use_trace = True

    def fit(self, X, y):
        """Compute CSP filters.
        
        Parameters
        ----------
        X : ndarray
            Covariance matrices, shape (n_trials, n_channels, n_channels).
        y : ndarray
            Labels, shape (n_trials,).
        
        Returns
        -------
        CSP object
            The CSP instance.
        """
        X = X.copy()
        y = y.copy()

        self.classes_ = np.unique(y)
        
        self.C_ = []
        for l in self.classes_:
            Cs = X[y==l]
            
            if self.use_trace:
                Cs /= np.trace(Cs, axis1=-2, axis2=-1)[:, np.newaxis, np.newaxis]

            C = np.mean(Cs, axis=0)
            self.C_.append(C)
        
        if len(self.classes_) != 2:
            raise(ValueError("CSP only support 2-class situation."))
        W, D, A = csp_kernel(self.C_[0], self.C_[1])
        
        ix = np.argsort(np.abs(D-0.5))[::-1]
        W = W[ix]
        D = D[ix]
        A = A[ix]

        self._set_filters(W[:self.n_filters])
        self.patterns_ = A[:self.n_filters]
        # self.eigenvalues_ = D[:self.n_filters]
        return self

    def transform(self, X):
        """Transform X to csp features.
        
        Parameters
        ----------
        X : ndarray
            Covariance matrices, shape (n_trials, n_channels, n_channels)
        
        Returns
        -------
        features : ndarray
            Log transformed CSP features, shape (n_trials, n_filters).
        """
        X = X.copy()
        X = np.reshape(X, (-1, *X.shape[-2:]))
        if self.use_trace:
            X /= np.trace(X, axis1=-2, axis2=-1)[:, np.newaxis, np.newaxis]
        Xf = super().transform(X)
        features = np.zeros((len(Xf), self.n_filters))
        for i, x in enumerate(Xf):
            features[i] = np.log(np.clip(np.diag(x), 1e-9, None)) # safe log
        return features


class SPoC(CSP):
    """SPoC.
    
    Parameters
    ----------
    n_filters : int, optional
        The number of SPoC filters to use (default 2).
    use_trace : bool, optional
        If True, use trace operation to normalize covariance matrices.
    
    """
    def fit(self, X, y):
        """Compute SPoC filters, the same as CSP except y is continuous variable.
        
        Parameters
        ----------
        X : ndarray
            Covariance matrices, shape (n_trials, n_channels, n_channels).
        y : ndarray
            Labels, shape (n_trials,).
        
        Returns
        -------
        SPoC object
            The SPoC instance.
        """
        X = X.copy().astype(np.float)
        weights = y.copy().astype(np.float)

        weights -= np.mean(weights)
        weights /= np.std(weights)

        if self.use_trace:
            X /= np.trace(X, axis1=-2, axis2=-1)[:, np.newaxis, np.newaxis]

        C = np.mean(X, axis=0)
        C = nearestPD(C)
        
        weight_X = np.zeros_like(X)
        for i, w in enumerate(weights):
            weight_X[i] = w*X[i]
        Cz = np.mean(weight_X, axis=0)

        # TODO: not read paper, direct copy from pyriemann, need verify
        D, W = eigh(Cz, C)
        D = np.abs(D.real)
        W = W.real
        ix = np.argsort(D)[::-1]
        D = D[ix]
        W = W[:, ix].T

        A = robustPattern(W.T, Cz)
        self._set_filters(W[:self.n_filters])
        self.patterns_ = A[:self.n_filters]
        # self.sig_filters_ = D[:self.n_filters]
        return self


class MultiCSP(CSP):
    """Multi-class CSP based on Wentrup's method [1]_.

    Parameters
    ----------
    n_filters : int, optional
        The number of CSP filters to use (default 2).
    use_trace : bool, optional
        If True, use trace operation to normalize covariance matrices.
    ajd_method : str, optional
        Approximate joint diagonalization method (default uwedge).

        Supported methods: `uwedge`, `rjd`, `ajd_pham`.

    Notes
    -----
    .. [1] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common spatial patterns and information theoretic feature extraction." Biomedical Engineering, IEEE Transactions on 55, no. 8 (2008): 1991-2000.
    """
    def __init__(self, n_filters=2, use_trace=True, ajd_method='uwedge'):
        super().__init__(n_filters=n_filters, use_trace=use_trace)   
        self.ajd_method = ajd_method
        
    def fit(self, X, y):
        """Compute CSP filters.
        
        Parameters
        ----------
        X : ndarray
            Covariance matrices, shape (n_trials, n_channels, n_channels).
        y : ndarray
            Labels, shape (n_trials,).
        
        Returns
        -------
        MultiCSP object
            The MultiCSP instance.
        """
        X = X.copy()
        y = y.copy()

        self.classes_ = np.unique(y)
        
        self.C_ = []
        for l in self.classes_:
            Cs = X[y==l]
            
            if self.use_trace:
                Cs /= np.trace(Cs, axis1=-2, axis2=-1)[:, np.newaxis, np.newaxis]

            C = np.mean(Cs, axis=0)
            self.C_.append(C)
        
        W, D = ajd(np.stack(self.C_), method=self.ajd_method)
        Ctot = np.mean(np.stack(self.C_), axis=0)

        W = W / np.sqrt(np.diag(np.dot(np.dot(W, Ctot), W.T)))[:, np.newaxis]

        mutual_info = []
        # class probability
        Pc = [np.mean(y == c) for c in self.classes_]
        for j in range(len(W)):
            a = 0
            b = 0
            for i in range(len(self.classes_)):
                tmp = np.dot(np.dot(W[j], self.C_[i]), W[j].T)
                a += Pc[i] * np.log(np.sqrt(tmp))
                b += Pc[i] * (tmp ** 2 - 1)
            mi = - (a + (3.0 / 16) * (b ** 2))
            mutual_info.append(mi)
        ix = np.argsort(mutual_info)[::-1]

        W = W[ix]
        A = pinv(W).T

        self._set_filters(W[:self.n_filters])
        self.patterns_ = A[:self.n_filters]
        return self


class FBCSP(FilterBank):
    """FilterBank CSP with mutual information selection.
    
    Parameters
    ----------
    n_filters : int, optional
        The number of CSP filters to use (default 2).
    n_infos : int, optional
        The number of mutual information to use (default 6).
    use_trace : bool, optional
        If True, use trace operation to normalize covariance matrices.
    
    """

    def __init__(self, n_filters=2, n_infos=6, use_trace=True):
        self.n_filters = n_filters
        self.n_infos = n_infos
        
        super().__init__(
            CSP(
                n_filters=n_filters, 
                use_trace=use_trace
            )
        )
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif, k=self.n_infos)

    def fit(self, X, y):
        super().fit(X, y=y)
        features = super().transform(X)
        features = features.transpose((1, 2, 0))
        features = np.reshape(features, (features.shape[0], -1))
        self.feature_selector.fit(features, y=y)
        return self

    def transform(self, X):
        features = super().transform(X)
        features = features.transpose((1, 2, 0))
        features = np.reshape(features, (features.shape[0], -1))
        features = self.feature_selector.transform(features)
        return features


    
