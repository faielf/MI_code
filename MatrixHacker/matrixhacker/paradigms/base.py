import time
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import mne
from mne.utils import verbose
from joblib import Parallel, delayed

from ..datasets.base import pick_channels


class BaseParadigm(metaclass=ABCMeta):
    """Abstract Base Paradigm.
    
    Parameters
    ----------
    channels : None|list of str, optional
        Selected channel names, if None use all channels in dataset.
    events : None|list of str, optional
        Selected event names, if None use all events in dataset.
    intervals : None|list of tuple of length 2, optional
        Selected event intervals, if None use default intervals in dataset. 
        If only one interval passed, all events use the same interval.
        Otherwise the number of tuples should be the same as the number of events.
    srate: None|float, optional
        Selected sampling rate, if None use default srate in dataset. Otherwise which resampling startegy should i use.
    """

    def __init__(self, 
        channels=None, 
        events=None, 
        intervals=None, 
        srate=None):
        """Init"""
        self.select_channels = None if channels is None else [ch_name.upper() for ch_name in channels]
        self.event_list = events
        self.intervals = intervals
        self.srate = srate
        self._raw_hook = None
        self._epochs_hook = None
        self._data_hook = None

    @abstractmethod
    def is_valid(self, dataset):
        """Verify the dataset is compatible with the paradigm.

        This method is called to verify dataset is compatible with the
        paradigm.

        This method should raise an error if the dataset is not compatible
        with the paradigm. This is for example the case if the
        dataset is an ERP dataset for motor imagery paradigm, or if the
        dataset does not contain any of the required events.

        Parameters
        ----------
        dataset : dataset instance
            The dataset to verify.
        """
        pass
    
    def _map_events_intervals(self, dataset):
        event_list = self.event_list
        intervals = self.intervals

        if event_list is None:
            # use all events in dataset
            event_list = list(dataset.event_info.keys())
        
        used_events = {ev: dataset.event_info[ev][0] for ev in event_list}

        if intervals is None:
            used_intervals = {ev: dataset.event_info[ev][1] for ev in event_list}
        elif len(intervals) == 1:
            used_intervals = {ev: intervals[0] for ev in event_list}
        else:
            if len(event_list) != len(intervals):
                raise ValueError("intervals should be the same number of events")
            used_intervals = {ev: interval for ev, interval in zip(event_list, intervals)}

        return used_events, used_intervals

    def register_raw_hook(self, hook):
        """Register raw hook before epoch operation.
        
        Parameters
        ----------
        hook : callable object
            Callable object to process Raw object before epoch operation.
            Its' signature should look like:

            hook(raw, caches) -> raw, caches

            where caches is an dict stroing information, raw is MNE Raw instance.
        """
        self._raw_hook = hook

    def register_epochs_hook(self, hook):
        """Register epochs hook after epoch operation.
        
        Parameters
        ----------
        hook : callable object
            Callable object to process Epochs object after epoch operation.
            Its' signature should look like:

            hook(epochs, caches) -> epochs, caches

            where caches is an dict storing information, epochs is MNE Epochs instance.
        """
        self._epochs_hook = hook

    def register_data_hook(self, hook):
        """Register data hook before return data.
        
        Parameters
        ----------
        hook : callable object
            Callable object to process ndarray data before return it.
            Its' signature should look like:

            hook(X, y, meta, caches) -> X, y, meta, caches

            where caches is an dict storing information, X, y are ndarray object, meta is a pandas DataFrame instance.
        """
        self._data_hook = hook

    def unregister_raw_hook(self):
        """Unregister raw hook before epoch operation.
        
        """
        self._raw_hook = None

    def unregister_epochs_hook(self):
        """Register epochs hook after epoch operation.
        
        """
        self._epochs_hook = None

    def unregister_data_hook(self):
        """Register data hook before return data.
        
        """
        self._data_hook = None

    @verbose
    def get_data(self, dataset, subjects=None, verbose=False):
        """Get data from dataset with selected subjects.
        
        Parameters
        ----------
        dataset : Dataset instance.
            Dataset.
        subjects : None|list of int, optional
            Selected subject ids, if None, use all subjects in dataset.
        verbose : bool, optional
            Print processing information.
        
        Returns
        -------
        Xs : dict
            An dict of selected events data X.
        ys : dict
            An dict of selected events label y.
        metas : dict
            An dict of selected events metainfo meta.
        """
        if not self.is_valid(dataset):
            raise TypeError(
                "Dataset {:s} is not valid for the current paradigm. Check your events and channels settings".format(dataset.code))

        data = dataset.get_data(subjects)

        # # events, interval checking
        used_events, used_intervals = self._map_events_intervals(dataset)

        Xs = {}
        ys = {}
        metas = {}
        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    # do raw hook
                    caches = {}
                    if self._raw_hook:
                        raw, caches = self._raw_hook(raw, caches)

                    # pick selected channels by order
                    channels = dataset.channels if self.select_channels is None else self.select_channels
                    # picks = mne.pick_channels(raw.ch_names, channels, ordered=True)

                    picks = pick_channels(raw.ch_names, channels, ordered=True)

                    # find available events, first check stim_channels then annotations
                    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
                    if len(stim_channels) > 0:
                        events = mne.find_events(raw, shortest_event=0, initial_event=True)
                    else:
                        # convert event_id to its number type instead of default auto-renaming in 0.19.2
                        events, _ = mne.events_from_annotations(raw, event_id=(lambda x: int(x)))

                    for event_name in used_events.keys():
                        # mne.pick_events returns any matching events in include
                        # only raise Runtime Error when nothing is found
                        # then we just skip this event
                        try:
                            selected_events = mne.pick_events(events, include=used_events[event_name])
                        except RuntimeError:
                            continue 

                        # transform Raw to Epochs

                        epochs = mne.Epochs(raw, selected_events,
                            event_id={event_name: used_events[event_name]},
                            event_repeated='drop', 
                            tmin=used_intervals[event_name][0],
                            tmax=used_intervals[event_name][1]-1./raw.info['sfreq'],
                            picks=picks,
                            proj=False, baseline=None, preload=True)

                        # do epochs hook
                        if self._epochs_hook:
                            epochs, caches = self._epochs_hook(epochs, caches)
                        
                        # FIXME: is this resample reasonable?
                        if self.srate:
                            # as MNE suggested, decimate after extract epochs
                            # low-pass raw object in raw_hook to prevent aliasing problem 
                            epochs = epochs.resample(self.srate)
                            # epochs = epochs.decimate(dataset.srate//self.srate)

                        # retrieve X, y and meta
                        X = epochs.get_data()
                        y = epochs.events[:, -1]
                        meta = pd.DataFrame(
                            {
                                "subject": [subject]*len(epochs),
                                "session": [session]*len(epochs),
                                "run": [run]*len(epochs), 
                            })

                        # do data hook
                        if self._data_hook:
                            X, y, meta, caches = self._data_hook(X, y, meta, caches)

                        # collecting data
                        pre_X = Xs.get(event_name)
                        if pre_X is not None:
                            Xs[event_name] = np.concatenate((pre_X, X), axis=0)
                        else:
                            Xs[event_name] = X

                        pre_y = ys.get(event_name)
                        if pre_y is not None:
                            ys[event_name] = np.concatenate((pre_y, y), axis=0)
                        else:
                            ys[event_name] = y

                        pre_meta = metas.get(event_name)
                        if pre_meta is not None:
                            metas[event_name] = pd.concat(
                                (pre_meta, meta),
                                axis=0,
                                ignore_index=True
                            )
                        else:
                            metas[event_name] = meta
                        
        return Xs, ys, metas

    def __str__(self):
        desc = "{}".format(self.__class__.__name__)
        return desc