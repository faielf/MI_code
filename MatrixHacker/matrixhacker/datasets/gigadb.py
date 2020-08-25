"""
GigaDb Motor imagery dataset.
"""

from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
from mne.utils import verbose
import numpy as np

from .base import BaseDataset, mne_data_path, loadmat


GIGA_URL = 'ftp://penguin.genomics.cn/pub/10.5524/100001_101000/100295/mat_data/'  # noqa


class Cho2017(BaseDataset):
    """Motor Imagery dataset from Cho et al 2017.

    Dataset from the paper [1]_.

    **Dataset Description**

    We conducted a BCI experiment for motor imagery movement (MI movement)
    of the left and right hands with 52 subjects (19 females, mean age ± SD
    age = 24.8 ± 3.86 years); Each subject took part in the same experiment,
    and subject ID was denoted and indexed as s1, s2, …, s52.
    Subjects s20 and s33 were both-handed, and the other 50 subjects
    were right-handed.

    EEG data were collected using 64 Ag/AgCl active electrodes.
    A 64-channel montage based on the international 10-10 system was used to
    record the EEG signals with 512 Hz sampling rates.
    The EEG device used in this experiment was the Biosemi ActiveTwo system.
    The BCI2000 system 3.0.2 was used to collect EEG data and present
    instructions (left hand or right hand MI). Furthermore, we recorded
    EMG as well as EEG simultaneously with the same system and sampling rate
    to check actual hand movements. Two EMG electrodes were attached to the
    flexor digitorum profundus and extensor digitorum on each arm.

    Subjects were asked to imagine the hand movement depending on the
    instruction given. Five or six runs were performed during the MI
    experiment. After each run, we calculated the classification
    accuracy over one run and gave the subject feedback to increase motivation.
    Between each run, a maximum 4-minute break was given depending on
    the subject's demands.

    References
    ----------

    .. [1] Cho, H., Ahn, M., Ahn, S., Kwon, M. and Jun, S.C., 2017.
           EEG datasets for motor imagery brain computer interface.
           GigaScience. https://doi.org/10.1093/gigascience/gix034
    """

    _EVENTS = {
        "left_hand": (1, (0, 3)), 
        "right_hand": (2, (0, 3)), 
    }

    _CHANNELS = [
        'FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
        'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
        'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ',
        'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ',
        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
        'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 
    ]

    def __init__(self):
        super().__init__(
            code='cho2017', 
            subjects=list(range(1, 53)), 
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=512, 
            paradigm='imagery'
        )

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        raw_mat = loadmat(self.data_path(subject)[0])['eeg']
        eeg_data_l = np.concatenate((raw_mat['imagery_left'], raw_mat['imagery_event'].reshape((1, -1))), axis=0)
        eeg_data_r = np.concatenate((raw_mat['imagery_right'], raw_mat['imagery_event'].reshape((1, -1))*2),
        axis=0)

        data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r])
        ch_names = [ch_name.upper() for ch_name in self._CHANNELS] + ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'STI 014']
        ch_types = ['eeg']*len(self._CHANNELS) + ['emg']*4 + ['stim']
        montage = make_standard_montage('standard_1005')
        # 0.19.2 can not guarante channel name case insensitive
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
        
        info = create_info(
            ch_names=ch_names, ch_types=ch_types, sfreq=self.srate, montage=montage)
        raw = RawArray(data=data, info=info, verbose=False)
        return {'session_0': {'run_0': raw}}

    @verbose
    def data_path(self, subject, path=None, force_update=False,
                  update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        dests = []
        url = '{u:s}s{s:02d}.mat'.format(u=GIGA_URL, s=subject)
        dests.append(mne_data_path(url, 'gigadb', path, force_update, update_path))
        return dests
  