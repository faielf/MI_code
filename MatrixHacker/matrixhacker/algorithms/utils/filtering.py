import numpy as np
from scipy import signal

from sklearn.base import BaseEstimator, TransformerMixin
from mne.filter import filter_data, construct_iir_filter, create_filter

def is_filter_stable(a):
    """Check if iir filter is stable, not for fir filters

    Parameters
    ----------
    a: ndarray
        a coefs from any iir filter

    Returns
    -------
    bool: bool
        True if stable
    """
    return np.all(np.abs(np.roots(a)))


class OnlineConsecutiveFilter(BaseEstimator, TransformerMixin):
    """Online consecutive filter.
    
    https://stackoverflow.com/questions/21862777/bandpass-butterworth-filter-frequencies-in-scipy
    """
    def __init__(self, srate, filters, return_sos=True):
        self.srate = srate
        self.filters = filters
        self.return_sos = return_sos
    
    def _compute_filter(self, l, h, fs, order=3, return_sos=True):
        """Compute butterworth filter coeffs.
        
        Parameters
        ----------
        l : float|None
            low cutoff frequency, if None return lowpass filter coeffs
        h : float|None
            high cutoff frequency, if None return highpass filter coeffs
        fs : float
            sampling rate
        return_sos : bool, optional
            if True, return sos coeffs, by default True
        
        Returns
        -------
        coeff:
            filter coeffs
        coeff_zi:
            zi coeffs used by lfilter
        """
        output = 'sos' if return_sos else 'ba'

        if h is None:
            coeff = signal.butter(order, l, 
                btype='highpass', output=output, fs=fs)
        elif l is None:
            coeff = signal.butter(order, h,
                btype='lowpass', output=output, fs=fs)
        else:
            coeff = signal.butter(order, [l, h], 
                btype='bandpass', output=output, fs=fs)
        
        if return_sos:
            coeff_zi = signal.sosfilt_zi(coeff)
        else:
            coeff_zi = signal.lfilter_zi(*coeff)

        return coeff, coeff_zi

    def fit(self, X, y=None):
        """Compatiable with sklearn , use X to scale zi
        X of shape (Ne, ) initial sample, X must be 1d
        """
        Ne = len(X)
        self._coeffs = []
        self._coeffs_zi = []
        for lfreq, hfreq in self.filters:
            coeffs, coeffs_zi = self._compute_filter(lfreq, hfreq, 
                self.srate, order=5, return_sos=self.return_sos)
            self._coeffs.append(coeffs)
            self._coeffs_zi.append([coeffs_zi.copy()*X[i] for i in range(Ne)])
        return self

    def _lfilter(self, x, coeffs, coeffs_zi, use_sos=True):
        if use_sos:
            y, zi = signal.sosfilt(coeffs, x, zi=coeffs_zi)
        else:
            y, zi = signal.lfilter(*coeffs, x, zi=coeffs_zi)
        return y, zi

    def transform(self, X):
        Ne = len(X)
        X_filt = np.zeros((len(self.filters), *X.shape))
        for i in range(len(self.filters)):
            for j in range(Ne):
                X_filt[i, j], self._coeffs_zi[i][j] = self._lfilter(
                    X[j], 
                    self._coeffs[i], 
                    self._coeffs_zi[i][j], 
                    use_sos=self.return_sos
                    )
        return X_filt


class OnlineBlockFilter(BaseEstimator, TransformerMixin):
    """Online block filter.
    
    https://stackoverflow.com/questions/21862777/bandpass-butterworth-filter-frequencies-in-scipy
    """
    def __init__(self, srate, filters, use_reflect=False, return_sos=True):
        self.srate = srate
        self.filters = filters
        self.use_reflect = use_reflect
        self.return_sos = return_sos
    
    def _compute_filter(self, l, h, fs, order=3, return_sos=True):
        """Compute butterworth filter coeffs.
        
        Parameters
        ----------
        l : float|None
            low cutoff frequency, if None return lowpass filter coeffs
        h : float|None
            high cutoff frequency, if None return highpass filter coeffs
        fs : float
            sampling rate
        order: int
            order of butterworth filter
        return_sos : bool, optional
            if True, return sos coeffs, by default True
        
        Returns
        -------
        coeff:
            filter coeffs
        coeff_zi:
            zi coeffs used by lfilter
        """
        output = 'sos' if return_sos else 'ba'

        if h is None:
            coeff = signal.butter(order, l, 
                btype='highpass', output=output, fs=fs)
        elif l is None:
            coeff = signal.butter(order, h,
                btype='lowpass', output=output, fs=fs)
        else:
            coeff = signal.butter(order, [l, h], 
                btype='bandpass', output=output, fs=fs)
        
        if return_sos:
            coeff_zi = signal.sosfilt_zi(coeff)
        else:
            coeff_zi = signal.lfilter_zi(*coeff)

        return coeff, coeff_zi

    def fit(self, X=None, y=None):
        """Compatiable with sklearn , use X to scale zi in each block
        """
        self._coeffs = []
        self._coeffs_zi = []
        for lfreq, hfreq in self.filters:
            coeffs, coeffs_zi = self._compute_filter(lfreq, hfreq, 
                self.srate, order=5, return_sos=self.return_sos)
            self._coeffs.append(coeffs)
            self._coeffs_zi.append(coeffs_zi)
        return self

    def _lfilter(self, x, coeffs, coeffs_zi, use_sos=True):
        if use_sos:
            y, zi = signal.sosfilt(coeffs, x, zi=coeffs_zi)
        else:
            y, zi = signal.lfilter(*coeffs, x, zi=coeffs_zi)
        return y, zi

    def _reflect(self, X):
        pre_X = X.copy()[...,::-1]
        double_X = np.concatenate((pre_X, X), axis=-1)
        return double_X

    def _unreflect(self, double_X):
        Ns = double_X.shape[-1]//2
        X = double_X[..., Ns:]
        return X

    def transform(self, X):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        if self.use_reflect:
            X = self._reflect(X)
        Nt, Ne, Ns = X.shape
        X_filt = np.zeros((len(self.filters), Nt, Ne, Ns))
        for i in range(len(self.filters)):
            for j in range(Nt):
                for k in range(Ne):
                    X_filt[i, j, k], _ = self._lfilter(
                        X[j, k], 
                        self._coeffs[i], 
                        self._coeffs_zi[i]*X[j, k, 0], 
                        use_sos=self.return_sos
                        )
        if self.use_reflect:
            X_filt = self._unreflect(X_filt)
        return X_filt


class BlockFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq, filters):
        self.sfreq = sfreq
        self.filters = filters

    def fit(self, X=None, y=None):
        self.iir_params = []
        iir_params = {
            "order": 4, 
            "ftype": 'butter', 
            'output': 'sos'
        }

        for band in self.filters:
            self.iir_params.append(
                construct_iir_filter(iir_params, f_pass=band, f_stop=None, sfreq=self.sfreq, btype='bandpass', verbose=False)
            )
        # for band in self.filters:
        #     self.hs_.append(
        #         create_filter(None, self.sfreq, band[0], band[1], 
        #         method='iir', iir_params=construct_iir_filter(iir_params, f_pass=band, f_stop=None, sfreq=sfreq, btype='bandpass')
        #         )
        #     )

        return self

    def transform(self, X):
        Xf = []
        for band, iir_params in zip(self.filters, self.iir_params):
            Xf.append(filter_data(X, self.sfreq, band[0], band[1], method='iir', iir_params=iir_params, verbose=False, n_jobs=-1))
        
        Xf = np.stack(Xf)
        return Xf
