import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel


def kernel(kernel_name, X1, X2=None, gamma=None):
    """Compute pairwise kernel of X1 and X2.
    
    Parameters
    ----------
    kernel_name : str
        linear | rbf | primal
    X1 : ndarray
        shape (n_trials1, n_features)
    X2 : ndarray
        shape (n_trials2, n_features)
    gamma: float
        gamma parameter for rbf_kernel

    Returns
    -------
    K : ndarray
        shape (n_trials1, n_trials2)
    """
    if kernel_name == 'linear':
        K = linear_kernel(X1, Y=X2)
    elif kernel_name == 'rbf':
        K = rbf_kernel(X1, Y=X2, gamma=gamma)
    else:
        K = X1.T
    
    return K


