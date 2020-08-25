"""Covariance estimation related methods.

**Covariance** is extremely important in BCI community. Many algorithms require different matrix decompostion methods of estimated covariances.
However, noisy and unstable characteristics of EEG signals make it hard to get numerically stable results. 
This problem can be alleviated with **shrinkage covariance estimation** methods, like `LedoitWolf`.

"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
from joblib import Parallel, delayed


def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator.
    
    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).
    
    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    C, _ = ledoit_wolf(X.T)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator.
    
    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).
    
    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    C, _ = oas(X.T)
    return C


def _cov(X):
    """Wrapper for sklearn sample covariance estimator.
    
    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).
    
    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    C = empirical_covariance(X.T)
    return C


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator.
    
    Parameters
    ----------
    X : ndarray
        EEG signal, shape (n_channels, n_samples).
    
    Returns
    -------
    C : ndarray
        Estimated covariance, shape (n_channels, n_channels).
    """
    _, C, _, _ = fast_mcd(X.T)
    return C


estimators = {
    'cov': _cov,
    'lwf': _lwf,
    'oas': _oas,
    'mcd': _mcd,
}


def _check_est(est):
    """Check if a given estimator is valid.

    Parameters
    ----------
    est : callable object or str
        Could be the name of estimator or a callable estimator itself.

    Returns
    -------
    est: callable object
        A callable estimator.
    """
    if callable(est):
        pass
    elif est in estimators.keys():
        est = estimators[est]
    else:
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function""" % (est, (' , ').join(estimators.keys())))
    return est


def covariances(X, estimator='cov', n_jobs=1):
    """Estimation of covariance matrix.
    
    Parameters
    ----------  
    X : ndarray
        EEG signal, shape (..., n_channels, n_samples).
    estimator : str or callable object, optional
        Covariance estimator to use (the default is `cov`, which uses empirical covariance estimator). For regularization, consider `lwf` or `oas`.

        **supported estimators**

            `cov`: empirial covariance estimator

            `lwf`: ledoit wolf covariance estimator

            `oas`: oracle approximating shrinkage covariance estimator

            `mcd`: minimum covariance determinant covariance estimator
    n_jobs : int or None, optional 
        The number of CPUs to use to do the computation (the default is 1, -1 for all processors).
    
    Returns
    -------
    covmats : ndarray
        covariance matrices, shape (..., n_channels, n_channels)

    See Also
    --------
    covariances_erp
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    shape = X.shape
    X = np.reshape(X, (-1, shape[-2], shape[-1]))

    parallel = Parallel(n_jobs=n_jobs)
    est = _check_est(estimator)
    covmats = parallel(
        delayed(est)(x) for x in X)

    covmats = np.reshape(covmats, (*shape[:-2], shape[-2], shape[-2]))    
    return covmats


def covariances_erp(X, P, estimator='cov', n_jobs=1):
    """Estimation of covariance matrix combined with erp signals.
    
    Parameters
    ----------  
    X : ndarray
        EEG signal, shape (n_trials, ..., n_channels, n_samples)
    P : ndarray
        ERP signal, shape (..., n_components, n_samples)
    estimator : str or callable object, optional
        Covariance estimator to use (the default is `cov`, which uses empirical covariance estimator). 
        For regularization, consider `lwf` or `oas`.

        **supported estimators**

            `cov`: empirial covariance estimator

            `lwf`: ledoit wolf covariance estimator

            `oas`: oracle approximating shrinkage covariance estimator

            `mcd`: minimum covariance determinant covariance estimator
    n_jobs : int or None, optional 
        The number of CPUs to use to do the computation (the default is 1, -1 for all processors).
    
    Returns
    -------
    covmats : ndarray
        covariance matrices, shape (..., n_channels, n_channels)

    Notes
    -----
    This method concatenates EEG **X** and ERP **P** along with channel dimension, 
    which results in a new matrix with `n_channels+n_components` channels. 
    The rest of computation is the same as `covariances` method.

    See Also
    --------
    covariances
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    P = np.asarray(P)
    P = np.atleast_2d(P)

    if X.ndim == 2:
        X = np.concatenate((X, P), axis=-2)
    else:
        P = np.repeat(np.expand_dims(P, axis=0), X.shape[0], axis=0)
        X = np.concatenate((X, P), axis=-2)

    shape = X.shape

    X = np.reshape(X, (-1, shape[-2], shape[-1]))

    parallel = Parallel(n_jobs=n_jobs)
    est = _check_est(estimator)
    covmats = parallel(
        delayed(est)(x) for x in X)

    covmats = np.reshape(covmats, (*shape[:-2], shape[-2], shape[-2]))    
    return covmats


class Covariance(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix.
    
    Parameters
    ----------
    estimator : str or callable object, optional
        Covariance estimator to use (the default is `cov`, which uses empirical covariance estimator). For regularization, consider `lwf` or `oas`.

        **supported estimators**

            `cov`: empirial covariance estimator

            `lwf`: ledoit wolf covariance estimator

            `oas`: oracle approximating shrinkage covariance estimator

            `mcd`: minimum covariance determinant covariance estimator
    n_jobs : int or None, optional 
        The number of CPUs to use to do the computation (the default is 1, -1 for all processors).
    
    See Also
    --------
    ERPCovariance
    """
    def __init__(self, estimator='cov', n_jobs=1):
        self.estimator= estimator
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        """Not used, only for compatibility with sklearn API.
        
        Parameters
        ----------
        X : ndarray
            EEG signal, shape (..., n_channels, n_samples).
        y : ndarray
            Labels.

        Returns
        -------
        self : Covariance instance
            The Covariance instance.
        """
        return self

    def transform(self, X):
        """Transform EEG to covariance matrix.
        
        Parameters
        ----------
        X : ndarray
            EEG signal, shape (..., n_channels, n_samples).
        
        Returns
        -------
        covmats : ndarray
            Estimated covariances, shape (..., n_channels, n_channels)
        """
        covmats = covariances(X, estimator=self.estimator, n_jobs=self.n_jobs)
        return covmats


class ERPCovariance(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix for ERP signal.

    Parameters
    ----------
    estimator : str or callable object, optional
        Covariance estimator to use (the default is `cov`, which uses empirical covariance estimator). For regularization, consider `lwf` or `oas`.

        **supported estimators**

            `cov`: empirial covariance estimator

            `lwf`: ledoit wolf covariance estimator

            `oas`: oracle approximating shrinkage covariance estimator

            `mcd`: minimum covariance determinant covariance estimator
    svd : None or int, optional
        If not None (default is None), the prototype responses will be reduced using SVD with the number of components passed in svd parameter.
    n_jobs : int or None, optional 
        The number of CPUs to use to do the computation (the default is 1, -1 for all processors).
    
    Attributes
    ----------
    classes : ndarray
        The label of each class, shape (n_classes,).
    P : ndarray
        Prototyped responses, shape (n_c, n_samples), 

        n_c equals to `n_channels*n_classes`, if svd is not None, n_c equals to `n_classes*svd`.

    Notes
    -----
    Estimation of special form covariance matrix dedicated to ERP processing.
    For each class, a prototyped response is obtained by average across trial :

    .. math::
        \mathbf{P} = \\frac{1}{N} \sum_i^N \mathbf{X}_i

    and a super trial is build using the concatenation of P and the trial X :

    .. math::
        \mathbf{\\tilde{X}}_i =  \left[
                                \\begin{array}{c}
                                \mathbf{X}_i \\\\
                                \mathbf{P}
                                \end{array}
                                \\right]

    This super trial :math:`\mathbf{\\tilde{X}}_i` will be used for covariance
    estimation.

    This allows to take into account the spatial structure of the ERP signal, as
    described in [1]_.

    See Also
    --------
    Covariance

    References
    ----------
    .. [1] Congedo, Marco, Alexandre Barachant, and Rajendra Bhatia. "Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review." Brain-Computer Interfaces 4.3 (2017): 155-174.
    """
    def __init__(self, estimator='cov', svd=None, n_jobs=1):
        self.classes = None
        self.estimator = estimator
        self.svd = svd
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Estimate the prototyped response for each class in label y.
        
        Parameters
        ----------
        X : ndarray
            EEG signal, shape (n_trials, n_channels, n_samples)
        y : ndarray
            Labels corresponding to each trial, shape (n_trials,)
        
        Returns
        -------
        self: ERPCovariance instance
            The ERPCovariance instance.
        """
        self.classes = np.unique(y)
        
        self.P = []
        for c in self.classes:
            # Prototyped responce for each class
            P = np.mean(X[y == c, ...], axis=0)

            # Apply svd if requested
            if self.svd is not None:
                U, s, V = np.linalg.svd(P)
                P = np.dot(U[:, 0:self.svd].T, P)

            self.P.append(P)

        self.P = np.concatenate(self.P, axis=-2)
        return self

    def transform(self, X):
        """Transform EEG signal X to special ERP covariance matrix with prototyped responses.
        
        Parameters
        ----------
        X : ndarray
            EEG signal, shape (n_trials, n_channels, n_samples)
        
        Returns
        -------
        covmats : ndarray
            Special ERP covariance matrix, shape (n_trials, n_channels+n_c, n_channels+n_c).
            
            n_c equals to `n_channels*n_classes`

            if svd is not None, n_c equals to `n_classes*svd`.
        """
        covmats = covariances_erp(X, self.P, estimator=self.estimator, n_jobs=self.n_jobs)
        return covmats


