'''
Zhou2016.
'''
import os

from mne import create_info
from mne.io import read_raw_cnt
from mne.channels import make_standard_montage
from mne.utils import verbose
import numpy as np

from .base import BaseDataset, mne_data_path, loadmat, upper_ch_names

ZHOU_URL = '/home/swolf/Data/Data/BCI/MNE-zhou-2016'


class Zhou2016(BaseDataset):
    """Motor Imagery dataset from Zhou et al 2016.

    Dataset from the article *A Fully Automated Trial Selection Method for
    Optimization of Motor Imagery Based Brain-Computer Interface* [1]_.
    This dataset contains data recorded on 4 subjects performing 3 type of
    motor imagery: left hand, right hand and feet.

    Every subject went through three sessions, each of which contained two
    consecutive runs with several minutes inter-run breaks, and each run
    comprised 75 trials (25 trials per class). The intervals between two
    sessions varied from several days to several months.

    A trial started by a short beep indicating 1 s preparation time,
    and followed by a red arrow pointing randomly to three directions (left,
    right, or bottom) lasting for 5 s and then presented a black screen for
    4 s. The subject was instructed to immediately perform the imagination
    tasks of the left hand, right hand or foot movement respectively according
    to the cue direction, and try to relax during the black screen.

    References
    ----------

    .. [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated
           Trial Selection Method for Optimization of Motor Imagery Based
           Brain-Computer Interface. PLoS ONE 11(9).
           https://doi.org/10.1371/journal.pone.0162657
    """
    _EVENTS = {
        "left_hand": (1, (1, 6)),
        "right_hand": (2, (1, 6)), 
        "feet": (3, (1, 6))
    }

    _CHANNELS = [
        'FP1', 'FP2', 'FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CPZ', 'CP4', 'O1', 'OZ', 'O2'
    ]

    def __init__(self):
        super().__init__(
            code='zhou2016', 
            subjects=list(range(1, 5)),
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=250,
            paradigm='imagery'
        )

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        """return data for a single subject"""
        sessions = {}
        sess_dests = self.data_path(subject)
        montage = make_standard_montage('standard_1005')
        # 0.19.2 can not guarante channel name case insensitive
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        for sess_id, run_dests in enumerate(sess_dests):
            runs = {}
            for run_id, run_dest in enumerate(run_dests):
                raw = read_raw_cnt(run_dest, eog=['VEOU', 'VEOL'], preload=True,stim_channel=False)
                raw = upper_ch_names(raw)

                raw.set_montage(montage)

                runs['run_%d' % run_id] = raw
            sessions['session_%d' % sess_id] = runs
        return sessions

    @verbose
    def data_path(self, subject, path=None, force_update=False,
                  update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))
        
        # FIXME: remote server url parser and downloading
        # dests = [mne_data_path(url, self.code, path, force_update, update_path)]

        sess_dests = []
        path = ZHOU_URL

        for sess_id in range(1, 4):
            run_dests = []
            url = 'subject_{:d}/{:d}A.cnt'.format(subject, sess_id)
            run_dests.append(os.path.join(path, url))
            url = 'subject_{:d}/{:d}B.cnt'.format(subject, sess_id)
            run_dests.append(os.path.join(path, url))
            sess_dests.append(run_dests)
        return sess_dests
