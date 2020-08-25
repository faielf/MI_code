"""Transfer Component Analysis.

Modified from https://github.com/jindongwang/transferlearning/blob/master/code/traditional/TCA/TCA.py
"""

import numpy as np
from scipy.linalg import eigh
from .base import kernel


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit_transform(self, Xs, Xt):
        ns, nt = len(Xs), len(Xt)
        X = np.concatenate((Xs, Xt), axis=0)
        n, n_feat = X.shape
        # X /= np.linalg.norm(X, axis=0) # why this step?
        e = np.concatenate((1/ns*np.ones((ns, 1)), -1/nt*np.ones((nt, 1))))
        M = e*e.T
        # M = M /np.linalg.norm(M, 'fro') # why this step?
        H = np.eye(n) - 1/(n) * np.ones((n, n))
        K =kernel(self.kernel_type, X, None, gamma=self.gamma)
        reg_term = np.eye(n_feat) if self.kernel_type == 'primal' else np.eye(n)
        a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * reg_term
        b = np.linalg.multi_dot([K, H, K.T])

        self.D, self.W = eigh(a, b)
        Z = np.dot(self.W[:, :self.dim].T, K)
        # Z /= np.linalg.norm(Z, axis=0) # why and why this step??
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new
