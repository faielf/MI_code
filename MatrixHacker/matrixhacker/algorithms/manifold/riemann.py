"""Most riemann methods are herited from Alexandre Barachant's pyRiemann package.

Some signatures are modified and simplified according to my personal needs.
If you prefer original methods, see https://github.com/alexandrebarachant/pyRiemann for more details.

"""
from functools import partial

import numpy as np
import autograd.numpy as anp
from scipy.linalg import eigvalsh, inv, eigh
from scipy.linalg import sqrtm as scipy_sqrtm
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax
from joblib import Parallel, delayed

from pymanopt.manifolds import Rotations
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

from ..utils.base import (sqrtm, invsqrtm, logm, expm, powm, whitenm, unwhitenm)


def logmap(Pi, P):
    """Logarithm map.

    Logarithm map projects :math:`\mathbf{P}_i \in \mathcal{M}` to the tangent space point 
    :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` at :math:`\mathbf{P} \in \mathcal{M}`.
    
    Parameters
    ----------
    Pi : ndarray
        SPD matrix.
    P : ndarray
        Reference point.
        
    Returns
    -------
    Si : ndarray
        Tangent space point (in matrix form).
    """
    P12 = sqrtm(P)
    iP12 = invsqrtm(P)
    wPi = iP12@Pi@iP12
    Si = P12@logm(wPi)@P12
    return Si


def expmap(Si, P):
    """Exponential map.

    Exponential map projects :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` bach to the manifold
    :math:`\mathcal{M}`.
    
    Parameters
    ----------
    Si : ndarray
        Tangent space point (in matrix form).
        
    P : ndarray
        Reference point.
    
    Returns
    -------
    Pi : ndarray
        SPD matrix.
    """
    P12 = sqrtm(P)
    iP12 = invsqrtm(P)
    wSi = iP12@Si@iP12
    Pi = P12@expm(wSi)@P12
    return Pi


def geodesic(P1, P2, t):
    """Geodesic.
    
    The geodesic curve between any two SPD matrices :math:`\mathbf{P}_1,\mathbf{P}_2 \in \mathcal{M}`.

    Parameters
    ----------
    P1 : ndarray
        SPD matrix.
    P2 : ndarray
        SPD matrix, the same shape of P1.
    t : float
        :math:`0 \leq t \leq 1`.
    
    Returns
    -------
    phi : ndarray
        SPD matrix on the geodesic curve between P1 and P2.
    """
    P12 = sqrtm(P1)
    iP12 = invsqrtm(P1)
    wP2 = iP12@P2@iP12
    phi = P12@powm(wP2, t)@P12
    return phi


def _get_sample_weight(sample_weight, N):
    """Get the sample weights.

    If none provided, weights init to 1. otherwise, weights are normalized.
    """
    if sample_weight is None:
        sample_weight = np.ones(N)
    if len(sample_weight) != N:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= np.sum(sample_weight)
    return sample_weight


def distance_riemann(A, B):
    """Riemannian distance between two covariance matrices A and B.

    Parameters
    ----------
    A : ndarray
        First positive-definite matrix, shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    B : ndarray
        Second positive-definite matrix.

    Returns
    -------
    ndarray | float
        Riemannian distance between A and B.

    Notes
    -----
    .. math::
            d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B.
    """
    if A.ndim == 2:
        dist = np.sqrt((np.log(eigvalsh(A, B))**2).sum())
    elif A.ndim == 3:
        dist = np.array([np.sqrt((np.log(eigvalsh(tmp, B))**2).sum()) for tmp in A])
    return dist


def mean_riemann(covmats, tol=1e-8, maxiter=100, init=None, sample_weight=None):
    """Return the mean covariance matrix according to the Riemannian metric.

    Parameters
    ----------
    covmats : ndarray
        Covariance matrices set, shape (n_trials, n_channels, n_channels).
    tol : float, optional
        The tolerance to stop the gradient descent (default 1e-8).
    maxiter : int, optional
        The maximum number of iteration (default 50).
    init : None|ndarray, optional
        A covariance matrix used to initialize the gradient descent (default None), if None the arithmetic mean is used.
    sample_weight : None|ndarray, optional
        The weight of each sample (efault None), if None weights are 1 otherwise weights are normalized.

    Returns
    -------
    C : ndarray
        The Riemannian mean covariance matrix.
    
    Notes
    -----
    The procedure is similar to a gradient descent minimizing the sum of riemannian distance to the mean.

    .. math::
        \mathbf{C} = \\arg \min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    where :math:\delta_R is riemann distance.
    """
    # init
    sample_weight = _get_sample_weight(sample_weight, len(covmats))
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = np.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = np.dot(np.dot(Cm12, covmats[index, :, :]), Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        C = np.dot(np.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def tangent_space(Pis, P):
    """Logarithm map projects SPD matrices to the tangent vectors.
    
    Parameters
    ----------
    Pis : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels).
    P : ndarray
        Reference point.
    
    Returns
    -------
    Sis : ndarray
        Tangent vectors, shape (n_trials, n_channels*(n_channels+1)/2).
    """
    n_trials, n_channels, n_channels = Pis.shape
    n_features = int(n_channels*(n_channels+1)/2)

    idx = np.triu_indices_from(P)
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) + np.eye(n_channels))[idx]
    Sis = np.zeros((n_trials, n_features))

    # P12 = sqrtm(P)
    iP12 = invsqrtm(P)
    for i, Pi in enumerate(Pis):
        wPi = iP12@Pi@iP12
        Si = logm(wPi)
        Sis[i] = np.multiply(coeffs, Si[idx])
    
    return Sis
    

def untangent_space(Sis, P):
    """Exponential map projects tangent vectors back to the SPD matrices.
    
    Parameters
    ----------
    Sis : ndarray
        Tangent vectors, shape (n_trials, n_channels*(n_channels+1)/2).
    P : ndarray
        Reference point.
    
    Returns
    -------
    Pis : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels).
    """
    n_trials, n_features = Sis.shape
    n_channels = int((np.sqrt(1 + 8 * n_features) - 1) / 2)
    
    idx = np.triu_indices_from(P)
    # didx = np.diag_indices(n_channels)
    Pis = np.zeros((n_trials, n_channels, n_channels))
    Pis[:, idx[0], idx[1]] = Sis

    P12 = sqrtm(P)
    # iP12 = invsqrtm(P)
    for i, Pi in enumerate(Pis):
        triuc = np.triu(Pi, 1) / np.sqrt(2)
        Pi = (np.diag(np.diag(Pi)) +  triuc + triuc.T)
        Pis[i] = P12@expm(Pi)@P12
    return Pis


def pt_tangent(A, B, Sbs):
    """Parallel transport in the tangent space.
    
    Parameters
    ----------
    A : ndarray
        SPD matrix.
    B : ndarray
        SPD matrix
    Sbs : ndarray
        The tangent vector (matrix form) in the tangent space of manifold B, 
        shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    
    Returns
    -------
    Sas : ndarray
        The tangent vector (matrix form) in the tangent space of manifold A.
    """
    Sbs = Sbs.reshape(-1, *Sbs.shape[-2:])
    n_trials, _, _ = Sbs.shape
    E = scipy_sqrtm(A@inv(B))

    Sas = np.zeros_like(Sbs)
    for i, Sb in enumerate(Sbs):
        Sas[i] = E@Sb@E.T

    if n_trials == 1:
        Sas = Sas[0]
    
    return Sas


def pt_manifold(A, B, Pbs):
    """Parallel transport in the manifold space.
    
    Parameters
    ----------
    A : ndarray
        SPD matrix.
    B : ndarray
        SPD matrix
    Pbs : ndarray
        SPD matrix in the manifold B, shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    
    Returns
    -------
    Pas : ndarray
        SPD matrix in the manifold A.
    """
    Pbs = Pbs.reshape(-1, *Pbs.shape[-2:])
    n_trials, _, _ = Pbs.shape
    E = scipy_sqrtm(A@inv(B))
    
    Pas = np.zeros_like(Pbs)
    for i, Pb in enumerate(Pbs):
        Pas[i] = E@Pb@E.T

    if n_trials == 1:
        Pas = Pas[0]

    return Pas


def recenter(C, M):
    """Re-center.
    
    Re-center :math:`\mathbr{C} \in \mathcal{M}` to the identity centroid.

    Parameters
    ----------
    C : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channles) or (n_channels, n_channels).
    M : ndarray
        The centroid of manifold.
    
    Returns
    -------
    Cr : ndarray
        Re-centered matrices.
    """
    C = C.reshape(-1, *C.shape[-2:])
    n_trials = len(C)

    Cr = np.zeros_like(C)

    iM12 = invsqrtm(M)
    for i, Ci in enumerate(C):
        Cr[i] = iM12@Ci@iM12

    if n_trials == 1:
        Cr = Cr[0]

    return Cr


def rescale(C, s, M=None):
    """Re-scale.
    
    Re-scale :math:`\mathbr{C} \in \mathcal{M}` by scaling factor s.

    Parameters
    ----------
    C : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channles) or (n_channels, n_channels).
    s : float
        Scaling factor.
    M : ndarray | None
        The centroid of manifold, defaults to the identity matrix.
    
    Returns
    -------
    Cs : ndarray
        Re-scaled matrices.
    """
    C = C.reshape(-1, *C.shape[-2:])
    n_trials, n_channels, _ = C.shape

    Cs = np.zeros_like(C)

    if M is not None:
        M12 = sqrtm(M)
        iM12 = invsqrtm(M)
    else:
        M12 = np.eye(n_channels)
        iM12 = np.eye(n_channels)
    
    for i, Ci in enumerate(C):
        Cs[i] = M12@powm(iM12@Ci@iM12, s)@M12
    
    if n_trials == 1:
        Cs = Cs[0]
    
    return Cs
        

def rotate(C, R):
    """Rotate.
    
    Rotate :math:`\mathbr{C} \in \mathcal{M}` with rotation matrix :math:`\mathbr{U}`.

    Parameters
    ----------
    C : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    U : ndarray
        Rotation matrix.
    
    Returns
    -------
    Cr : ndarray
        Rotated matrices.
    """
    C = C.reshape(-1, *C.shape[-2:])
    n_trials, n_channels, _ = C.shape

    Cr = np.zeros_like(C)

    for i, Ci in enumerate(C):
        Cr[i] = R@Ci@R.T
    
    if n_trials == 1:
        Cr = Cr[0]
    
    return Cr


def get_scale_factor(Cs, Ct):
    """Get scalefactor in rescale step, transform Cs to Ct.
    
    Parameters
    ----------
    Cs : ndarray
        source covariance matrices, shape (n_trials, n_channels, n_channels)
    Ct : ndarray
        target covariance matrices, shape (n_trials, n_channels, n_channels)

    Returns
    -------
    s : float
        scale factor
    """
    Ms = mean_riemann(Cs)
    Mt = mean_riemann(Ct)

    ds = np.sum(np.square(distance_riemann(Cs, Ms)))
    dt = np.sum(np.square(distance_riemann(Ct, Mt)))

    s = np.sqrt(dt/ds)
    return s


def _procruster_cost_function_euc(R, Mt, Ms):
    weights = anp.ones(len(Mt))

    c = []
    for Mti, Msi in zip(Mt, Ms):
        t1 = Msi
        t2 = anp.dot(R, anp.dot(Mti, R.T))
        ci = anp.linalg.norm(t1-t2)**2
        c.append(ci)
    c = anp.array(c)

    return anp.dot(c, weights)


def _procruster_cost_function_rie(R, Mt, Ms):
    weights = anp.ones(len(Mt))

    c = []
    for Mti, Msi in zip(Mt, Ms):
        t1 = Msi
        t2 = anp.dot(R, anp.dot(Mti, R.T))
        ci = distance_riemann(t1, t2)**2
        c.append(ci)
    c = anp.array(c)

    return anp.dot(c, weights)


def _procruster_egrad_function_rie(R, Mt, Ms):
    weights = anp.ones(len(Mt))
    
    g = []
    for Mti, Msi, wi in zip(Mt, Ms, weights):
        iMti12 = invsqrtm(Mti)
        Msi12 = sqrtm(Msi)
        term_aux = anp.dot(R, anp.dot(Msi, R.T))
        term_aux = anp.dot(iMti12, anp.dot(term_aux, iMti12))
        gi = 4 * anp.dot(anp.dot(iMti12, logm(term_aux)), anp.dot(Msi12, R))
        g.append(gi * wi)

    g = anp.sum(g, axis=0)

    return g


def get_rotation_matrix(Mt, Ms, metric='euc'):
    Mt = Mt.reshape(-1, *Mt.shape[-2:])
    Ms = Ms.reshape(-1, *Ms.shape[-2:])

    n = Mt[0].shape[0]
    manifolds = Rotations(n)

    if metric == 'euc':
        cost = partial(_procruster_cost_function_euc, Mt=Mt, Ms=Ms)  
        problem = Problem(manifold=manifolds, cost=cost, verbosity=0)
    elif metric == 'rie':
        cost = partial(_procruster_cost_function_rie, Mt=Mt, Ms=Ms)    
        egrad = partial(_procruster_egrad_function_rie, Mt=Mt, Ms=Ms) 
        problem = Problem(manifold=manifolds, cost=cost, egrad=egrad, verbosity=0) 

    solver = SteepestDescent(mingradnorm=1e-3)

    Ropt = solver.solve(problem)

    return Ropt


class TangentSpace(BaseEstimator, TransformerMixin):
    """Tangent space projection.

    Attributes
    ----------
    reference_ : ndarray
        If fit, the reference point for tangent space mapping.
    """

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None, sample_weight=None):
        """Estimate the reference point.

        Parameters
        ----------
        X : ndarray
            SPD matrices, shape (n_trials, n_channels,n_channels).
        y : None|ndarray, optional
            Not used, here for compatibility with sklearn API.
        sample_weight : None|ndarray, optional
            Weight of each trial (default None). If None provided, weights init to 1, otherwise, weights are normalized.

        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        # compute mean covariance
        self.reference_ = mean_riemann(X, sample_weight=sample_weight)
        return self

    def _check_data_dim(self, X):
        """Check data shape and return the size of cov mat."""
        shape_X = X.shape
        if len(X.shape) == 2:
            Ne = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if Ne != int(Ne):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(Ne)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')

    def transform(self, X):
        """Tangent space projection.

        Parameters
        ----------
        X : ndarray
            SPD matrices, shape (n_trials, n_channels, n_channels).

        Returns
        -------
        ts : ndarray
            The tangent space projection of the matrices, shape (n_trials, n_channels*(n_channels+1)/2).
        """
        X = X.copy()
        self._check_reference_points(X)
        return tangent_space(X, self.reference_)

    def inverse_transform(self, X, y=None):
        """Inverse transform.

        Project back a set of tangent space vector in the manifold.

        Parameters
        ----------
        X : ndarray
            SPD matrices, shape (n_trials, n_channels*(n_channels+1)/2).
        y : None|ndarray, optional
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        cov : ndarray
            The covariance matrices corresponding to each of tangent vector, shape (n_trials, n_channels, n_channels).
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        self._check_reference_points(X)
        return untangent_space(X, self.reference_)


class MDM(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.
    """

    def __init__(self, use_trace=True, n_jobs=1):
        """Init."""
        # store params for cloning purpose
        self.use_trace = use_trace
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        X = X.copy()
        y = y.copy()

        self.classes_ = np.unique(y)

        self.covmeans_ = []

        if self.use_trace:
            X /= np.trace(X, axis1=-2, axis2=-1)[:, np.newaxis, np.newaxis]

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.n_jobs == 1:
            for l in self.classes_:
                self.covmeans_.append(
                    mean_riemann(X[y == l], sample_weight=sample_weight[y == l]))
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_riemann)(X[y == l], sample_weight=sample_weight[y == l])
                for l in self.classes_)

        return self

    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        covtest = covtest.copy()
        covtest = np.reshape(covtest, (-1, *covtest.shape[-2:]))

        if self.use_trace:
            covtest /= np.trace(covtest, axis1=-2, axis2=-1)[:, np.newaxis, np.newaxis]

        Nc = len(self.covmeans_)

        if self.n_jobs == 1:
            dist = [distance_riemann(covtest, self.covmeans_[m])
                    for m in range(Nc)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance_riemann)(
                covtest, self.covmeans_[m])
                for m in range(Nc))

        dist = np.stack(dist)
        return dist.T

    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-1*self._predict_distances(X))


class FGDA(BaseEstimator, TransformerMixin):

    """Fisher Geodesic Discriminant analysis.

    Project data in Tangent space, apply a FLDA to reduce dimention, and
    project filtered data back in the manifold.
    For a complete description of the algorithm, see [1]

    Parameters
    ----------
    metric : string (default: 'riemann')
        The type of metric used for reference point mean estimation.
        see `mean_covariance` for the list of supported metric.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.

    See Also
    --------
    FgMDM
    TangentSpace

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self):
        """Init."""
        pass

    def _fit_lda(self, X, y, sample_weight=None):
        """Helper to fit LDA."""
        self.classes_ = np.unique(y)
        self._lda = LinearDiscriminantAnalysis(n_components=len(self.classes_) - 1,
                        solver='lsqr',
                        shrinkage='auto')

        ts = self._ts.fit_transform(X, sample_weight=sample_weight)
        self._lda.fit(ts, y)

        W = self._lda.coef_.copy()
        self._W = np.dot(
            np.dot(W.T, np.linalg.pinv(np.dot(W, W.T))), W)
        return ts

    def _retro_project(self, ts):
        """Helper to project back in the manifold."""
        ts = np.dot(ts, self._W)
        return self._ts.inverse_transform(ts)

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the reference point and the FLDA.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray 
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        self : FGDA instance
            The FGDA instance.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        self._ts = TangentSpace()
        self._fit_lda(X, y, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """Filtering operation.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        covs : ndarray, shape (n_trials, n_channels, n_channels)
            covariances matrices after filtering.
        """
        X = X.copy()
        ts = self._ts.transform(X)
        return self._retro_project(ts)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        covs : ndarray, shape (n_trials, n_channels, n_channels)
            covariances matrices after filtering.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        self._ts = TangentSpace()
        ts = self._fit_lda(X, y, sample_weight=sample_weight)
        return self._retro_project(ts)


class FgMDM(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Classification by Minimum Distance to Mean with geodesic filtering.

    Apply geodesic filtering described in [1], and classify using MDM algorithm
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular mdm.
    This is basically a pipeline of FGDA and MDM

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    See Also
    --------
    MDM
    FGDA
    TangentSpace

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, n_jobs=1):
        """Init."""
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit FgMDM.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self._mdm = MDM(n_jobs=self.n_jobs)
        self._fgda = FGDA()
        cov = self._fgda.fit_transform(X, y)
        self._mdm.fit(cov, y)
        return self

    def predict(self, X):
        """get the predictions after FDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)

    def transform(self, X):
        """get the distance to each centroid after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
            the distance to each centroid according to the metric.
        """
        cov = self._fgda.transform(X)
        return self._mdm.transform(cov)


class TSclassifier(BaseEstimator, ClassifierMixin):

    """Classification in the tangent space.

    Project data in the tangent space and apply a classifier on the projected
    data. This is a simple helper to pipeline the tangent space projection and
    a classifier. Default classifier is LogisticRegression

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    clf: sklearn classifier (default LogisticRegression)
        The classifier to apply in the tangent space

    See Also
    --------
    TangentSpace

    Notes
    -----
    .. versionadded:: 0.2.4
    """

    def __init__(self, clf=LogisticRegression(solver='lbfgs', multi_class='auto')):
        """Init."""
        self.clf = clf

    def fit(self, X, y):
        """Fit TSclassifier.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : TSclassifier. instance
            The TSclassifier. instance.
        """
        ts = TangentSpace()
        self._pipe = make_pipeline(ts, self.clf)
        self._pipe.fit(X, y)
        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of ifloat, shape (n_trials, n_classes)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict_proba(X)


class Potato(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Artefact detection with the Riemannian Potato.

    The Riemannian Potato [1] is a clustering method used to detect artifact in
    EEG signals. The algorithm iteratively estimate the centroid of clean
    signal by rejecting every trial that have a distance greater than several
    standard deviation from it.

    Parameters
    ----------
    metric : string (default 'riemann')
        The type of metric used for centroid and distance estimation.
    threshold : int (default 3)
        The number of standard deviation to reject artifacts.
    n_iter_max : int (default 100)
        The maximum number of iteration to reach convergence.
    pos_label: int (default 1)
        The positive label corresponding to clean data
    neg_label: int (default 0)
        The negative label corresponding to artifact data

    Notes
    -----
    .. versionadded:: 0.2.3

    See Also
    --------
    Kmeans
    MDM

    References
    ----------
    [1] A. Barachant, A. Andreev and M. Congedo, "The Riemannian Potato: an
    automatic and adaptive artifact detection method for online experiments
    using Riemannian geometry", in Proceedings of TOBI Workshop IV, p. 19-20,
    2013.
    """

    def __init__(self, threshold=3, n_iter_max=100,
                 pos_label=1, neg_label=0):
        """Init."""
        self.threshold = threshold
        self.n_iter_max = n_iter_max
        if pos_label == neg_label:
            raise(ValueError("Positive and Negative labels must be different"))
        self.pos_label = pos_label
        self.neg_label = neg_label

    def fit(self, X, y=None):
        """Fit the potato from covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Potato instance
            The Potato instance.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()

        self._mdm = MDM()

        if y is not None:
            if len(y) != len(X):
                raise ValueError('y must be the same lenght of X')

            classes = np.int32(np.unique(y))

            if len(classes) > 2:
                raise ValueError('number of classes must be maximum 2')

            if self.pos_label not in classes:
                raise ValueError('y must contain a positive class')

            y_old = np.int32(np.array(y) == self.pos_label)
        else:
            y_old = np.ones(len(X))
        # start loop
        for n_iter in range(self.n_iter_max):
            ix = (y_old == 1)
            self._mdm.fit(X[ix], y_old[ix])
            y = np.zeros(len(X))
            d = np.squeeze(np.log(self._mdm.transform(X[ix])))
            self._mean = np.mean(d)
            self._std = np.std(d)
            y[ix] = self._get_z_score(d) < self.threshold

            if np.array_equal(y, y_old):
                break
            else:
                y_old = y
        return self

    def transform(self, X):
        """return the normalized log-distance to the centroid (z-score).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        z : ndarray, shape (n_epochs, 1)
            the normalized log-distance to the centroid.
        """
        d = np.squeeze(np.log(self._mdm.transform(X)))
        z = self._get_z_score(d)
        return z

    def predict(self, X):
        """predict artefact from data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of bool, shape (n_epochs, 1)
            the artefact detection. True if the trial is clean, and False if
            the trial contain an artefact.
        """
        z = self.transform(X)
        pred = z < self.threshold
        out = np.zeros_like(z) + self.neg_label
        out[pred] = self.pos_label
        return out

    def _get_z_score(self, d):
        """get z score from distance."""
        z = (d - self._mean) / self._std
        return z


class RecursiveRiemannMean(BaseEstimator, TransformerMixin):
    """Recursive Riemannian Mean Update.
    
    Parameters
    ----------
    init_M: ndarray 
        The initialization of M, shape (n_channels, n_channels).
    count: int
        The number of accumulated step.
    
    """

    def __init__(self, init_M=None, count=0):
        self.M = init_M
        self.count = count

    def fit(self, X, y=None):
        if self.M is None:
            self.M = X
        else:
            self.M = geodesic(self.M, X, 1/(self.count + 1))
            self.count = self.count + 1
        return self

    def transform(self, X=None):
        return self.M


