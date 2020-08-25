"""2019 China BCI Competition datasets.
"""
import os.path as op

import numpy as np
import mne
from mne.utils import verbose
from .base import BaseDataset, mne_data_path, loadmat

CBCIC2019001_URL = '/home/swolf/Data/Data/BCI/MNE-cnbcic-data/CBCIC2019001'
CBCIC2019004_URL = '/home/swolf/Data/Data/BCI/MNE-cnbcic-data/CBCIC2019004'

class CBCIC2019001(BaseDataset):
    """2019 China BCI competition Dataset for MI in preliminary contest.

    Motor imagery dataset from China BCI competition in 2019.

    This dataset contains EEG recordings from 18 subjects, performing 2 or 3 tasks 
    of motor imagery (left hand, right hand or feet). Data have been recored at 1000hz 
    with 64 electrodes (59 in use except ECG, HEOR, HEOL, VEOU, VEOL channels) by
    an neuracle EEG amplifier.

    """

    _EVENTS = {
        "left_hand": (1, (1.5, 5.5)),
        "right_hand": (2, (1.5, 5.5)),
        "feet": (3, (1.5, 5.5)),
        "open_eye_relax": (7, (0, 60)),
        "close_eye_relax": (8, (0, 60))
    }

    _CHANNELS = [
        'FPZ','FP1','FP2','AF3','AF4','AF7','AF8',
        'FZ','F1','F2','F3','F4','F5','F6','F7','F8',
        'FCZ','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8',
        'CZ','C1','C2','C3','C4','C5','C6',
        'T7','T8','CP1','CP2','CP3','CP4','CP5','CP6',
        'TP7','TP8','PZ','P3','P4','P5','P6','P7','P8',
        'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
        'OZ','O1','O2'
    ]
    
    def __init__(self):
        super().__init__(
            code="cbcic2019001",
            subjects=list(range(1, 19)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=1000,
            paradigm="imagery"
        )
        self.type_list = ['B']*5 + ['T'] + ['B']*7 + ['T']*2 + ['B']*2 + ['T']
        
    @verbose
    def data_path(self, subject, path=None, force_update=False, 
            update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise ValueError('Invalid subject {:d} given'.format(subject))
        # TODO: hardcoded data path
        path = CBCIC2019001_URL
        url = "{s:02d}/{t:s}{s:02d}01T.mat".format(t=self.type_list[subject-1], s=subject)
        dests = op.join(path, url)
        return dests

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        raw_mat = loadmat(self.data_path(subject))['EEG']
        srate = raw_mat['srate']

        data = raw_mat['data']
        data = data[:-5]
        stim = np.zeros((1, data.shape[-1]))
        for event in raw_mat['event']:
            stim[0, int(event['latency'])-1] = int(event['type'])
        data = np.concatenate((data, stim), axis=0)

        # cause mne channel pick bug, always use big-case forcibly
        ch_names = [chanloc['labels'] for chanloc in raw_mat['chanlocs']]
        ch_names = ch_names[:-5]
        ch_names = [ch_name.upper() for ch_name in ch_names]

        ch_types = ['eeg']*len(ch_names)
        ch_names = ch_names + ['STI 014']
        ch_types = ch_types + ['stim']
        montage = mne.channels.make_standard_montage('standard_1005')
        # 0.19.2 has no guarantee about case insensitive channel names
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        info = mne.create_info(
            ch_names, srate, 
            ch_types=ch_types, montage=montage)

        raw = mne.io.RawArray(data, info)
        return {"session_0": {"run_0": raw}}


class CBCIC2019004(BaseDataset):
    _EVENTS = {
        "left_hand": (1, (0, 4)),
        "right_hand": (2, (0, 4)),
    }

    _CHANNELS = [
        'FPZ','FP1','FP2','AF3','AF4','AF7','AF8',
        'FZ','F1','F2','F3','F4','F5','F6','F7','F8',
        'FCZ','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8',
        'CZ','C1','C2','C3','C4','C5','C6',
        'T7','T8','CP1','CP2','CP3','CP4','CP5','CP6',
        'TP7','TP8','PZ','P3','P4','P5','P6','P7','P8',
        'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
        'OZ','O1','O2'
    ]
    
    def __init__(self):
        super().__init__(
            code="cbcic2019004",
            subjects=list(range(1, 7)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm="imagery"
        )
        
    @verbose
    def data_path(self, subject, path=None, force_update=False, 
            update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise ValueError('Invalid subject {:d} given'.format(subject))
        # TODO: hardcoded data path
        path = CBCIC2019004_URL
        dests = []
        for block_id in range(1, 5):
            url = "{s:02d}/block{b:d}.mat".format(s=subject, b=block_id)
            dests.append(op.join(path, url))
        return dests

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        dests = self.data_path(subject)

        runs = {}
        for i_run, dest in enumerate(dests):
            raw_mat = loadmat(dest)
            data = raw_mat['data']
            events = data[-1]
            data = data[:-6]

            stim = np.zeros((1, data.shape[-1]))
            for ie, event in enumerate(events):
                if event in [1, 2]:
                    stim[0, ie] = int(event)
            data = np.concatenate((data, stim), axis=0)

            # cause mne channel pick bug, always use big-case forcibly
            ch_names = self._CHANNELS

            ch_types = ['eeg']*len(ch_names)
            ch_names = ch_names + ['STI 014']
            ch_types = ch_types + ['stim']
            montage = mne.channels.make_standard_montage('standard_1005')
            # 0.19.2 has no guarantee about case insensitive channel names
            montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
            
            info = mne.create_info(
                ch_names, 250, 
                ch_types=ch_types, montage=montage)

            raw = mne.io.RawArray(data, info)
            runs["run_{:d}".format(i_run)] = raw
        return {"session_0": runs}

@DeprecationWarning
class CnbcicSSVEP2019(BaseDataset):
    _freq_list = [round(8+0.2*i, 2) for i in range(40)]
    _phase_list = [round((0+0.5*i)%2, 2) for i in range(40)]

    _FREQS, _PHASES, _EVENTS = {}, {}, {}
    for code, ix in enumerate(range(0, len(_freq_list), 4)):
        for inc in range(4):
            _FREQS["code_{:d}".format(code+inc*10+1)] = _freq_list[ix+inc]
            _PHASES["code_{:d}".format(code+inc*10+1)] = _phase_list[ix+inc]
            _EVENTS["code_{:d}".format(code+inc*10+1)] = (code+inc*10+1, (0, 3))

    _CHANNELS = [
        'FPZ','FP1','FP2','AF3','AF4','AF7','AF8',
        'FZ','F1','F2','F3','F4','F5','F6','F7','F8',
        'FCZ','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8',
        'CZ','C1','C2','C3','C4','C5','C6',
        'T7','T8','CP1','CP2','CP3','CP4','CP5','CP6',
        'TP7','TP8','PZ','P3','P4','P5','P6','P7','P8',
        'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
        'OZ','O1','O2'
    ]
    
    def __init__(self):
        super().__init__(
            code="2019cnbcic_ssvep",
            subjects=list(range(1, 61)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm="ssvep"
        )
        self.freqs = self._FREQS
        self.phases = self._PHASES
        
    @verbose
    def data_path(self, subject, path=None, force_update=False, 
            update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise ValueError('Invalid subject {:d} given'.format(subject))
        # TODO: hardcoded data path
        path = '/home/swolf/Data/Data/BCI/cnbcic2019/ssvep/AB/'
        destinations = []
        for i in range(1, 4):
            url = "S{:02d}/block{}.mat".format(subject, i)
            destinations.append(op.join(path, url))
        return destinations

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        paths = self.data_path(subject)
        runs = {}
        for i, path in enumerate(paths):
            raw_mat = loadmat(path)['EEG']
            srate = raw_mat['srate']

            data = raw_mat['data']
            data = data[:-5]
            stim = np.zeros((1, data.shape[-1]))
            for event in raw_mat['event']:
                stim[0, int(event['latency'])-1] = int(event['type'])
            data = np.concatenate((data, stim), axis=0)

            # cause mne channel pick bug, always use big-case forcibly
            ch_names = [chanloc['labels'] for chanloc in raw_mat['chanlocs']]
            ch_names = ch_names[:-5]
            ch_names = [ch_name.upper() for ch_name in ch_names]

            ch_types = ['eeg']*len(ch_names)
            ch_names = ch_names + ['STI 014']
            ch_types = ch_types + ['stim']
            montage = mne.channels.read_montage(
                'standard_1005',
                ch_names=ch_names)

            info = mne.create_info(
                ch_names, srate, 
                ch_types=ch_types, montage=montage)

            raw = mne.io.RawArray(data, info)
            runs["run_{:d}".format(i)] = raw
        return {"session_0": runs}

@DeprecationWarning
class CnbcicDrySSVEP2019(BaseDataset):
    _freq_list = [round(9.25+0.5*i, 2) for i in range(10)]
    _phase_list = [round((0+0.5*i)%2, 2) for i in range(10)]

    _FREQS, _PHASES, _EVENTS = {}, {}, {}
    for i in range(0, len(_freq_list)):
            _FREQS["code_{:d}".format(i+1)] = _freq_list[i]
            _PHASES["code_{:d}".format(i+1)] = _phase_list[i]
            _EVENTS["code_{:d}".format(i+1)] = (i+1, (0, 5))

    _CHANNELS = [
        'P3','C3','F3','FZ','F4','C4','P4', 
        'CZ', 'PZ', 
        'A1', 'FP1', 'FP2', 'T3', 'T5', 'O1', 'O2', 'F7', 'F8', 'A2', 
        'T6', 'T4'
    ]
    
    def __init__(self):
        super().__init__(
            code="2019cnbcic_dryssvep",
            subjects=list(range(1, 25)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=300,
            paradigm="ssvep"
        )
        self.freqs = self._FREQS
        self.phases = self._PHASES
        
    @verbose
    def data_path(self, subject, path=None, force_update=False, 
            update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise ValueError('Invalid subject {:d} given'.format(subject))
        # TODO: hardcoded data path
        path = '/home/swolf/Data/Data/BCI/cnbcic2019/ssvep/AB_dry/'
        destinations = []
        for i in range(1, 11):
            url = "S{:02d}/block{}.mat".format(subject, i)
            destinations.append(op.join(path, url))
        return destinations

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        paths = self.data_path(subject)
        runs = {}
        for i, path in enumerate(paths):
            raw_mat = loadmat(path)['EEG']
            srate = raw_mat['srate']

            data = raw_mat['data']
            ix = list(range(16)) + list(range(18, 20)) + list(range(21, 24))
            data = data[ix]
            stim = np.zeros((1, data.shape[-1]))
            for event in raw_mat['event']:
                stim[0, int(event['latency'])-1] = int(event['type'])
            data = np.concatenate((data, stim), axis=0)

            # cause mne channel pick bug, always use big-case forcibly
            ch_names = [ch_name.upper() for ch_name in self._CHANNELS]

            ch_types = ['eeg']*len(ch_names)
            ch_names = ch_names + ['STI 014']
            ch_types = ch_types + ['stim']
            montage = mne.channels.read_montage(
                'standard_1005',
                ch_names=ch_names)

            info = mne.create_info(
                ch_names, srate, 
                ch_types=ch_types, montage=montage)

            raw = mne.io.RawArray(data, info)
            runs["run_{:d}".format(i)] = raw
        return {"session_0": runs}
      
@DeprecationWarning
class CnbcicP3002019(BaseDataset):
    _EVENTS = {
        'target': (1, (0, 0.5)),
        'nontarget': (2, (0, 0.5))
    }

    _CODE_LABELS = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            _CODE_LABELS[i, j] = 70+i*10+j

    _CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4',
        'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
        'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
        'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
        'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
        'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
        'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
        'CB1','O1','OZ','O2','CB2'
    ]
    
    def __init__(self):
        super().__init__(
            code="2019cnbcic_p300",
            subjects=list(range(1, 61)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=1000,
            paradigm="p300"
        )
        self.code_labels = self._CODE_LABELS

        
    @verbose
    def data_path(self, subject, path=None, force_update=False, 
            update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise ValueError('Invalid subject {:d} given'.format(subject))
        # TODO: hardcoded data path
        path = '/home/swolf/Data/Data/BCI/cnbcic2019/p300/AB/'
        url = "S{:02d}/Train.mat".format(subject)
        destination = op.join(path, url)
        return destination


    def _map_label_to_rc_event(self, cur_label):
        r, c = np.where(self.code_labels==cur_label)
        ture_event = np.concatenate((c+1, r+7))
        return ture_event.tolist()


    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        runs = {}
        raw_mats = loadmat(self.data_path(subject))
        for i in range(1, 6):
            raw_mat = raw_mats['Char{}'.format(i)]
            data = raw_mat['Data']

            stim = np.zeros((1, data.shape[-1]))
            rcs = None
            for event_id, latency in zip(raw_mat['Event']['type'], raw_mat['Event']['latency']):
                if event_id in self.code_labels:
                    rcs = self._map_label_to_rc_event(event_id)
                else:
                    if rcs is not None:
                        if event_id in rcs:
                            stim[0, int(latency)-1] = int(self.event_info['target'][0])
                        else:
                            stim[0, int(latency)-1] = int(self.event_info['nontarget'][0])

            data = np.concatenate((data, stim), axis=0)

            # cause mne channel pick bug, always use big-case forcibly
            ch_names = [chanloc['labels'] for chanloc in raw_mat['Chanlocs']]
            ch_names = [ch_name.upper() for ch_name in ch_names]

            ch_types = ['eeg']*len(ch_names)
            ch_names = ch_names + ['STI 014']
            ch_types = ch_types + ['stim']
            montage = mne.channels.read_montage(
                'standard_1005',
                ch_names=ch_names)

            info = mne.create_info(
                ch_names, self.srate, 
                ch_types=ch_types, montage=montage)

            raw = mne.io.RawArray(data, info)
            runs["run_{:d}".format(i)] = raw
        return {"session_0": runs}


