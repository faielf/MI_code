'''
Munchi MI Deprecated.
https://doi.org/10.1371/journal.pone.0114853
'''

from mne import create_info
from mne.io import read_raw_eeglab
from mne.utils import verbose
import numpy as np

from .base import BaseDataset, mne_data_path, loadmat

MUNICH_URL = 'https://zenodo.org/record/1217449/files/'


class MunichMI(BaseDataset):
    """Munich Motor Imagery dataset.

    Motor imagery dataset from Grosse-Wentrup et al. 2009 [1]_.

    A trial started with the central display of a white fixation cross. After 3
    s, a white arrow was superimposed on the fixation cross, either pointing to
    the left or the right.
    Subjects were instructed to perform haptic motor imagery of the
    left or the right hand during display of the arrow, as indicated by the
    direction of the arrow. After another 7 s, the arrow was removed,
    indicating the end of the trial and start of the next trial. While subjects
    were explicitly instructed to perform haptic motor imagery with the
    specified hand, i.e., to imagine feeling instead of visualizing how their
    hands moved, the exact choice of which type of imaginary movement, i.e.,
    moving the fingers up and down, gripping an object, etc., was left
    unspecified.
    A total of 150 trials per condition were carried out by each subject,
    with trials presented in pseudorandomized order.

    Ten healthy subjects (S1–S10) participated in the experimental
    evaluation. Of these, two were females, eight were right handed, and their
    average age was 25.6 years with a standard deviation of 2.5 years. Subject
    S3 had already participated twice in a BCI experiment, while all other
    subjects were naive to BCIs. EEG was recorded at M=128 electrodes placed
    according to the extended 10–20 system. Data were recorded at 500 Hz with
    electrode Cz as reference. Four BrainAmp amplifiers were used for this
    purpose, using a temporal analog high-pass filter with a time constant of
    10 s. The data were re-referenced to common average reference
    offline. Electrode impedances were below 10 kΩ for all electrodes and
    subjects at the beginning of each recording session. No trials were
    rejected and no artifact correction was performed. For each subject, the
    locations of the 128 electrodes were measured in three dimensions using a
    Zebris ultrasound tracking system and stored for further offline analysis.


    References
    ----------
    .. [1] Grosse-Wentrup, Moritz, et al. "Beamforming in noninvasive
           brain–computer interfaces." IEEE Transactions on Biomedical
           Engineering 56.4 (2009): 1209-1219.

    """

    _EVENTS = {
        "left_hand": (1, (3, 10)),
        "right_hand": (2, (3, 10)), 
    }

    _CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
        'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
        'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
        'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
        'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    
    def __init__(self):
        super().__init__(
            code='weibo2014', 
            subjects=list(range(1, 11)),
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=500,
            paradigm='imagery'
        )

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        """return data for a single subject"""
        # FIXME: no channel names!!
        raw = read_raw_eeglab(self.data_path(subject)[0])

        return {"session_0": {"run_0": raw}}

    @verbose
    def data_path(self, subject, path=None, force_update=False,
                  update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        url = '{:s}subject{:d}.fdt'.format(MUNICH_URL, subject)
        mne_data_path(url, 'munichmi', path, force_update, update_path, verbose)
        
        url = '{:s}subject{:d}.set'.format(MUNICH_URL, subject)
        dests = [
            mne_data_path(url, 'munichmi', path, force_update, update_path, verbose)
        ]
        return dests