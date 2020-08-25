# -*- coding: utf-8 -*-
"""Joint Distribution Adaptation.

Modified from https://github.com/jindongwang/transferlearning/tree/master/code/traditional/JDA
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from .base import kernel


class JDA:
    def __init__(self, 
        kernel_type='linear', dim=30, lamb=1, gamma=1, T=10, clf=None, verbose=False):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T # number of iterations
        if clf is None:
            # self.clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            self.clf = KNeighborsClassifier(n_neighbors=1)
        else: 
            self.clf = clf # classifier to predict pesudo label
        self.verbose = verbose

    def fit_transform(self, Xs, ys, Xt, yt=None):
        ns, nt = len(Xs), len(Xt)
        X = np.concatenate((Xs, Xt), axis=0)
        n, n_feat = X.shape
        labels = np.unique(ys)

        # X /= np.linalg.norm(X, axis=0) # why this step?
        e = np.concatenate((1/ns*np.ones((ns, 1)), -1/nt*np.ones((nt, 1))))
        M0 = np.dot(e, e.T)
        
        H = np.eye(n) - 1/(n) * np.ones((n, n))

        pseudo_yt = None
        for t in range(self.T):
            Mc = 0

            if pseudo_yt is not None:
                for c in labels:
                    es = np.zeros((ns, 1))
                    et = np.zeros((nt, 1))
                    sc_ix = ys == c
                    tc_ix = pseudo_yt == c
                    es[sc_ix] = 1 / np.sum(sc_ix)
                    et[tc_ix] = -1 / np.sum(tc_ix)
                    e = np.concatenate((es, et), axis=0)
                    e[np.isinf(e)] = 0 # avoiding 0 number of c
                    Mc += np.dot(e, e.T)

            M = M0 + Mc
            # M /= np.linalg.norm(M, 'fro') # why this step?
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            reg_term = np.eye(n_feat) if self.kernel_type == 'primal' else np.eye(n)
            a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * reg_term
            b = np.linalg.multi_dot([K, H, K.T])
            self.D, self.W = eigh(a, b)

            Z = np.dot(self.W[:, :self.dim].T, K)
            # Z /= np.linalg.norm(Z, axis=0) # why and why this step??
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            pseudo_yt = self.clf.fit(Xs_new, ys).predict(Xt_new)
            if self.verbose:
                print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, np.mean(yt==pseudo_yt)))

        return Xs_new, Xt_new



