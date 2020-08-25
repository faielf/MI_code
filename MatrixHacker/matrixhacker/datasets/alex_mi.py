# -*- coding: utf-8 -*-
"""
Alex Motor imagery dataset.
"""

from .base import BaseDataset, mne_data_path, upper_ch_names
from mne.io import Raw
from mne.utils import verbose


ALEX_URL = 'https://zenodo.org/record/806023/files/'


class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset.

    Motor imagery dataset from the PhD dissertation of A. Barachant [1]_.

    This Dataset contains EEG recordings from 8 subjects, performing 2 task of
    motor imagination (right hand, feet or rest). Data have been recorded at
    512Hz with 16 wet electrodes (Fpz, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
    P7, P3, Pz, P4, P8) with a g.tec g.USBamp EEG amplifier.

    File are provided in MNE raw file format. A stimulation channel encoding
    the timing of the motor imagination. The start of a trial is encoded as 1,
    then the actual start of the motor imagination is encoded with 2 for
    imagination of a right hand movement, 3 for imagination of both feet
    movement and 4 with a rest trial.

    The duration of each trial is 3 second. There is 20 trial of each class.

    references
    ----------
    .. [1] Barachant, A., 2012. Commande robuste d'un effecteur par une
           interface cerveau machine EEG asynchrone (Doctoral dissertation,
           Université de Grenoble).
           https://tel.archives-ouvertes.fr/tel-01196752

    """
    
    _EVENTS = {
        "right_hand": (2, (0, 3)),
        "feet": (3, (0, 3)),
        "rest": (4, (0, 3))
    }

    _CHANNELS = [
        'FPZ','F7','F3','FZ','F4','F8',
        'T7','C3','C4','T8',
        'P7','P3','PZ','P4','P8'
    ]
    
    def __init__(self):
        super().__init__(
            code='alexeeg', 
            subjects=list(range(1, 9)),
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=512,
            paradigm='imagery'
        )

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        """return data for a single subject"""
        raw = Raw(self.data_path(subject)[0], preload=True)
        raw = upper_ch_names(raw)
        return {"session_0": {"run_0": raw}}

    @verbose
    def data_path(self, subject, path=None, force_update=False,
                  update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))
        url = '{:s}subject{:d}.raw.fif'.format(ALEX_URL, subject)
        dests = [mne_data_path(url, self.code, path, force_update, update_path)]
        return dests
