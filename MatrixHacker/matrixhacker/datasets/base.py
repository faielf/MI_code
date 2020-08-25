import os, time
import shutil
import os.path as op
from abc import ABCMeta, abstractmethod
from urllib import parse, request

import numpy as np
import scipy.io as sio
import h5py
import mne
from mne.datasets.utils import _get_path, _do_path_update
from mne.utils import _fetch_file, _url_to_local_path, verbose


def url_to_local_path(url, local_folder=''):
    """Mirror a url path in a local destination (keeping folder structure)."""
    destination = parse.urlparse(url).path
    # First char should be '/', and it needs to be discarded
    if len(destination) < 2 or destination[0] != '/':
        raise ValueError('Invalid URL')
    destination = os.path.join(local_folder, request.url2pathname(destination)[1:])
    return destination


def fetch(datastream, destination, resume=True):
    tmp_file = destination + ".part"

    if not os.path.exists(tmp_file):
        resume = False

    initial_size = 0
    if resume:
        with open(tmp_file, 'rb', buffering=0) as local_file:
            local_file.seek(0, 2)
            initial_size = local_file.tell()
        del local_file

    mode = 'ab' if initial_size > 0 else 'wb'
    chunk_size = 8192
    with open(tmp_file, mode) as local_file:
        while True:
            t0 = time.time()
            chunk = datastream.read(chunk_size)
            dt = time.time() - t0
            if dt < 5e-3:
                chunk_size *= 2
            elif dt > 0.1 and chunk_size > 8192:
                chunk_size = chunk_size // 2
            
            # reach the end of downloading file
            if not chunk:
                break
            local_file.write(chunk)

    shutil.move(tmp_file, destination)
        

@verbose
def mne_data_path(url, sign, path=None, force_update=False, update_path=True,
              verbose=None):
    """Get path to local copy of given dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote dataset

    Parameters
    ----------
    url : str
        Path to remote location of data
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the BNCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_(signifier)_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    """  # noqa: E501
    sign = sign.upper()
    key = 'MNE_DATASETS_{:s}_PATH'.format(sign)
    key_dest = 'MNE-{:s}-data'.format(sign.lower())
    path = _get_path(path, key, sign)
    destination = _url_to_local_path(url, op.join(path, key_dest))
    # Fetch the file
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False)

    # Offer to update the path
    _do_path_update(path, update_path, key, sign)
    return destination


def loadmat(filename):
    """Wrapper of scipy loadmat, works for  mat after matlab v7.3.
    """
    try:
        data = _loadmat(filename)
    except:
        data = _load_h5_matlab_version(filename)
    return data


def _loadmat(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    Notes: only works for mat before matlab v7.3
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _tolist(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        if ndarray.dtype == np.object:
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list
        else:
            return ndarray

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def squeeze(data):
    """Wrapper of numpy squeeze function , return scalar if data is scalar array.

    """
    data = np.squeeze(data)
    if data.shape == ():
        data = data.item()
    return data


def _load_h5_matlab_version(filename):
    '''
    this function should be called instead of direct h5py
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    Notes: only works for mat after matlab v7.3
    '''
    def _to_data(d, f):
        if isinstance(d, h5py.File) or isinstance(d, h5py.Group):
            data = {}
            for key in d:
                data[key] = _to_data(d[key], f)
            return data
        elif isinstance(d, h5py.Dataset):
            dtype = d.attrs['MATLAB_class'].decode('utf-8')
            if dtype == 'char':
                return _char_to_string(d, f)
            elif dtype == 'double':
                return _double_to_array(d, f)
            elif dtype == 'cell':
                return _cell_to_list(d, f)
            else:
                return None
        elif isinstance(d, h5py.Reference):
            data = _ref_to_data(d, f)
            return data
        else:
            return d
        
    def _cell_to_list(d, f):
        data = []
        for celld in d:
            if isinstance(celld, list):
                data.append(_cell_to_list(celld, f))
            else:
                data.append(_to_data(squeeze(celld), f))
        return data

    def _ref_to_data(d, f):
        data = _to_data(f[d], f)
        return data

    def _double_to_array(d, f):
        return squeeze(np.array(d))

    def _char_to_string(d, f):
        return ''.join([chr(c) for c in d])

    with h5py.File(filename) as f:
        data = _to_data(f, f)
    return data


def upper_ch_names(raw):
    raw.info['ch_names'] = [ch_name.upper() for ch_name in raw.info['ch_names']]
    for i, ch in enumerate(raw.info['chs']):
        ch['ch_name'] = raw.info['ch_names'][i]
    return raw


def pick_channels(ch_names, pick_chs, ordered=True, match_case='auto'):
    if match_case == 'auto':
        if len(set([ch_name.lower() for ch_name in ch_names])) < len(set(ch_names)):
            match_case = True
        else:
            match_case = False

    if match_case:
        picks = mne.pick_channels(ch_names, pick_chs, ordered=ordered)
    else:
        ch_names = [ch_name.lower() for ch_name in ch_names]
        pick_chs = [pick_ch.lower() for pick_ch in pick_chs]
        picks = mne.pick_channels(ch_names, pick_chs, ordered=ordered)

    return picks
    

class BaseDataset(metaclass=ABCMeta):
    """BaseDataset

    Parameters required for all datasets.

    parameters
    ----------
    subjects: List of int
        List of subject number

    sessions_per_subject: int
        Number of sessions per subject

    events: dict of strings, value of (event_id, interval) format
        String codes for events matched with labels in the stim channel.
        Currently avaliable codes can include:
        - left_hand
        - right_hand
        - hands
        - feet
        - rest
        - left_hand_right_foot
        - right_hand_left_foot
        - tongue
        - navigation
        - subtraction
        - word_ass (for word association)

    code: string
        Unique identifier for dataset, used in all plots

    channels: list of string
        availabe channels

    paradigm: ['p300','imagery', 'ssvep']
        Defines what sort of dataset this is (currently only imagery is
        implemented)
    """
    def __init__(self, 
        code, subjects, events, channels, srate, paradigm):
        if not isinstance(subjects, list):
            raise(ValueError("subjects must be a list"))
        if not isinstance(events, dict):
            raise(ValueError("events must be an dict"))
        if not isinstance(channels, list):
            raise(ValueError("channels must be a list of string"))

        self.code = code
        self.subject_list = subjects
        self.event_info = events
        self.channels = [ch.upper() for ch in channels]
        self.srate = srate
        self.paradigm = paradigm

    @abstractmethod
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        """Get path to local copy of a subject data.

        Parameters
        ----------
        subject : int
            Number of subject to use
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see :func:`mne.verbose`).

        Returns
        -------
        path : list of str
            Local path to the given data file. This path is contained inside a
            list of length one, for compatibility.
        """
        pass

    @abstractmethod
    def _get_single_subject_data(self, subject, verbose=None):
        """Return the data of a single subject.

        The returned data is a dictionary with the following structure

        data = {'session_id':
                    {'run_id': raw}
                }

        parameters
        ----------
        subject: int
            subject number

        returns
        -------
        data: Dict
            dict containing the raw data
        """
        pass

    def get_data(self, subjects=None):
        if subjects is None:
            subjects = self.subject_list
        
        if not isinstance(subjects, list):
            raise(ValueError("subjects must be a list"))
        
        data = dict()
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError('Invalid subject {:d} given'.format(subject))
            data[subject] = self._get_single_subject_data(subject)
        return data

    def __str__(self):
        event_info = '\n'.join(["    {}: {}".format(event_name, self.event_info[event_name]) for event_name in self.event_info])
        desc = """Dataset {:s}:\n  Subjects  {:d}\n  Srate     {:.1f}\n  Events   \n{}\n  Channels  {:d}\n""".format(
            self.code, 
            len(self.subject_list), 
            self.srate, 
            event_info, 
            len(self.channels)
        )
        return desc

    def __repr__(self):
        return self.__str__()


