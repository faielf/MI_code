"""Famous Physionet.
"""
import mne
import numpy as np

from mne.io import read_raw_edf, RawArray
from mne.channels import make_standard_montage
from mne.utils import verbose
from mne.datasets import eegbci

from .base import BaseDataset, upper_ch_names


BASE_URL = 'http://www.physionet.org/pn4/eegmmidb/'


class BasePhysionet(BaseDataset):
    """Physionet Motor Imagery dataset.

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org).
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
       The subject opens and closes the corresponding fist until the target
       disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
       The subject imagines opening and closing the corresponding fist until
       the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
       The subject opens and closes either both fists (if the target is on top)
       or both feet (if the target is on the bottom) until the target
       disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
       The subject imagines opening and closing either both fists
       (if the target is on top) or both feet (if the target is on the bottom)
       until the target disappears. Then the subject relaxes.

    parameters
    ----------

    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.

    executed: bool (default False)
        if True, return runs corresponding to motor execution.

    references
    ----------

    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
           Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
           interface (BCI) system. IEEE Transactions on biomedical engineering,
           51(6), pp.1034-1043.

    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
           P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
           H.E. and PhysioBank, P., PhysioNet: components of a new research
           resource for complex physiologic signals Circulation 2000 Volume
           101 Issue 23 pp. E215–E220.
    """

    _EVENTS = {
        "rest": (1, (0, 3)), 
        "left_hand": (2, (0, 3)), 
        "right_hand": (3, (0, 3)), 
        "hands": (4, (0, 3)), 
        "feet": (5, (0, 3)),
        "baseline_eyes_open": (6, (0, 60)), 
        "baseline_eyes_close": (7, (0, 60))
    }

    _CHANNELS = [
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 
        'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 
        'FP1', 'FPZ', 'FP2', 
        'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FT8', 
        'T7', 'T8', 'T9', 'T10', 
        'TP7', 'TP8', 
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
        'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 
        'O1', 'OZ', 'O2', 
        'IZ'    
    ]

    def __init__(self, paradigm, imagined=True, executed=False):
        super().__init__(
            code='eegbci', 
            subjects=list(range(1, 110)), 
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=160, 
            paradigm=paradigm
        )

        self.baseline_runs = [1, 2]
        self.feet_runs = []
        self.hand_runs = []

        if imagined:
            self.feet_runs += [6, 10, 14]
            self.hand_runs += [4, 8, 12]

        if executed:
            self.feet_runs += [5, 9, 13]
            self.hand_runs += [3, 7, 11]

    @verbose
    def _load_one_run(self, subject, run, preload=True, verbose=None):
        raw_fname = eegbci.load_data(subject, runs=[run],
                                     base_url=BASE_URL)[0]
        raw = read_raw_edf(raw_fname, preload=preload)
        raw.rename_channels(lambda x: x.strip('.'))
        montage = make_standard_montage('standard_1005')
        # 0.19.2 can not guarante channel name case insensitive
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
        raw = upper_ch_names(raw)
        raw.set_montage(montage)

        # creat simulate stim channel
        # mne >= 0.18
        events, _ = mne.events_from_annotations(raw)
        stim_channel = np.zeros((1, raw.n_times))
        for event in events:
            stim_channel[0, event[0]] = event[2]
        info = mne.create_info(['STI 014'], raw.info['sfreq'], ch_types=['stim'])
        raw = raw.add_channels([RawArray(stim_channel, info)], force_update_info=True)
        return raw

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        """return data for a single subject"""
        data = {}

        # baseline runs
        raw = self._load_one_run(subject, 1)
        stim = raw._data[-1]
        raw._data[-1, stim == 1] = 6
        data['run_1'] = raw

        raw = self._load_one_run(subject, 2)
        stim = raw._data[-1]
        raw._data[-1, stim == 1] = 7
        data['run_2'] = raw        

        # hand runs
        for run in self.hand_runs:
            data['run_%d' % run] = self._load_one_run(subject, run)

        # feet runs
        for run in self.feet_runs:
            raw = self._load_one_run(subject, run)

            # modify stim channels to match new event ids. for feets runs,
            # hand = 2 modified to 4, and feet = 3, modified to 5
            stim = raw._data[-1]
            raw._data[-1, stim == 2] = 4
            raw._data[-1, stim == 3] = 5
            data['run_%d' % run] = raw

        return {"session_0": data}

    @verbose
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        paths = eegbci.load_data(subject,
                                 runs=[1, 2] + self.hand_runs + self.feet_runs,
                                 base_url=BASE_URL)
        return paths


class PhysionetMI(BasePhysionet):
    """Physionet Motor Imagery dataset for MI.

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org).
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
       The subject opens and closes the corresponding fist until the target
       disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
       The subject imagines opening and closing the corresponding fist until
       the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
       The subject opens and closes either both fists (if the target is on top)
       or both feet (if the target is on the bottom) until the target
       disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
       The subject imagines opening and closing either both fists
       (if the target is on top) or both feet (if the target is on the bottom)
       until the target disappears. Then the subject relaxes.

    parameters
    ----------

    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.

    executed: bool (default False)
        if True, return runs corresponding to motor execution.

    references
    ----------

    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
           Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
           interface (BCI) system. IEEE Transactions on biomedical engineering,
           51(6), pp.1034-1043.

    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
           P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
           H.E. and PhysioBank, P., PhysioNet: components of a new research
           resource for complex physiologic signals Circulation 2000 Volume
           101 Issue 23 pp. E215–E220.
    """

    def __init__(self):
        super().__init__('imagery', imagined=True, executed=False)


class PhysionetME(BasePhysionet):
    """Physionet Motor Imagery dataset for motor execution.

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org).
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
       The subject opens and closes the corresponding fist until the target
       disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
       The subject imagines opening and closing the corresponding fist until
       the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
       The subject opens and closes either both fists (if the target is on top)
       or both feet (if the target is on the bottom) until the target
       disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
       The subject imagines opening and closing either both fists
       (if the target is on top) or both feet (if the target is on the bottom)
       until the target disappears. Then the subject relaxes.

    parameters
    ----------

    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.

    executed: bool (default False)
        if True, return runs corresponding to motor execution.

    references
    ----------

    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
           Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
           interface (BCI) system. IEEE Transactions on biomedical engineering,
           51(6), pp.1034-1043.

    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
           P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
           H.E. and PhysioBank, P., PhysioNet: components of a new research
           resource for complex physiologic signals Circulation 2000 Volume
           101 Issue 23 pp. E215–E220.
    """

    def __init__(self):
        super().__init__('movement', imagined=False, executed=True)
        
