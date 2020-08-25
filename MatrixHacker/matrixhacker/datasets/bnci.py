"""
Brain/Neuro Computer Interface (BNCI) datasets.
"""

from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
from mne.utils import verbose
import numpy as np

from .base import BaseDataset, mne_data_path, loadmat

BNCI_URL = 'http://bnci-horizon-2020.eu/database/data-sets/'
BBCI_URL = 'http://doc.ml.tu-berlin.de/bbci/'


class BNCI2014001(BaseDataset):
    """BNCI 2014-001 Motor Imagery dataset.

    Dataset IIa from BCI Competition 4 [1]_.

    **Dataset Description**

    This data set consists of EEG data from 9 subjects.  The cue-based BCI
    paradigm consisted of four different motor imagery tasks, namely the imag-
    ination of movement of the left hand (class 1), right hand (class 2), both
    feet (class 3), and tongue (class 4).  Two sessions on different days were
    recorded for each subject.  Each session is comprised of 6 runs separated
    by short breaks.  One run consists of 48 trials (12 for each of the four
    possible classes), yielding a total of 288 trials per session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
    were used to record the EEG; the montage is shown in Figure 3 left.  All
    signals were recorded monopolarly with the left mastoid serving as
    reference and the right mastoid as ground. The signals were sampled with.
    250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
    the amplifier was set to 100 μV . An additional 50 Hz notch filter was
    enabled to suppress line noise

    References
    ----------

    .. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
           Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
           and Nolte, G., 2012. Review of the BCI competition IV.
           Frontiers in neuroscience, 6, p.55.
    """

    _EVENTS = {
        "left_hand": (1, (2, 6)), 
        "right_hand": (2, (2, 6)), 
        "feet": (3, (2, 6)), 
        "tongue": (4, (2, 6))
    }

    _CHANNELS = [
        'FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 
        'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 
        'P1', 'PZ', 'P2', 'POZ'
    ]

    def __init__(self):
        super().__init__(
            code='bnci2014001', 
            subjects=list(range(1, 10)), 
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=250, 
            paradigm='imagery'
        )

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        """return data for a single subject"""
        dests = self.data_path(subject)
        sessions = {}
        for i_session, session in enumerate(('T', 'E')):
            runs = {}
            run_arrays = loadmat(dests[i_session])['data']
            for i_run, run_array in enumerate(run_arrays):
                X = run_array['X'].T
                trial = run_array['trial']
                y = run_array['y']
                stim =  np.zeros((1, X.shape[-1]))

                if y.size > 0:
                    stim[0, trial-1] = y

                data = np.concatenate((X, stim), axis=0)

                ch_names = [ch_name.upper() for ch_name in self._CHANNELS] + ['EOG1', 'EOG2', 'EOG3']
                ch_types = ['eeg']*len(self._CHANNELS) + ['eog']*3
                ch_names = ch_names + ['STI 014']
                ch_types = ch_types + ['stim']
                montage = make_standard_montage('standard_1005')
                # 0.19.2 has no guarantee about case insensitive channel names
                montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

                info = create_info(
                    ch_names, self.srate, 
                    ch_types=ch_types, montage=montage
                )

                raw = RawArray(data, info)
                runs['run_%d' % i_run] = raw
            sessions['session_%d' % i_session] = runs
        return sessions

    @verbose
    def data_path(self, subject, path=None, force_update=False,
                  update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        dests = []
        for run in ['T', 'E']:
            url = '{u:s}001-2014/A{s:02d}{r:s}.mat'.format(u=BNCI_URL, s=subject, r=run)
            dests.append(
                mne_data_path(url, 'bnci', path, force_update, update_path)
            )
        return dests


class BNCI2014004(BaseDataset):
    """BNCI 2014-004 Motor Imagery dataset.

    Dataset B from BCI Competition 2008.

    **Dataset description**

    This data set consists of EEG data from 9 subjects of a study published in
    [1]_. The subjects were right-handed, had normal or corrected-to-normal
    vision and were paid for participating in the experiments.
    All volunteers were sitting in an armchair, watching a flat screen monitor
    placed approximately 1 m away at eye level. For each subject 5 sessions
    are provided, whereby the first two sessions contain training data without
    feedback (screening), and the last three sessions were recorded with
    feedback.

    Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling
    frequency of 250 Hz.They were bandpass- filtered between 0.5 Hz and 100 Hz,
    and a notch filter at 50 Hz was enabled.  The placement of the three
    bipolar recordings (large or small distances, more anterior or posterior)
    were slightly different for each subject (for more details see [1]).
    The electrode position Fz served as EEG ground. In addition to the EEG
    channels, the electrooculogram (EOG) was recorded with three monopolar
    electrodes.

    The cue-based screening paradigm consisted of two classes,
    namely the motor imagery (MI) of left hand (class 1) and right hand
    (class 2).
    Each subject participated in two screening sessions without feedback
    recorded on two different days within two weeks.
    Each session consisted of six runs with ten trials each and two classes of
    imagery.  This resulted in 20 trials per run and 120 trials per session.
    Data of 120 repetitions of each MI class were available for each person in
    total.  Prior to the first motor im- agery training the subject executed
    and imagined different movements for each body part and selected the one
    which they could imagine best (e. g., squeezing a ball or pulling a brake).

    Each trial started with a fixation cross and an additional short acoustic
    warning tone (1 kHz, 70 ms).  Some seconds later a visual cue was presented
    for 1.25 seconds.  Afterwards the subjects had to imagine the corresponding
    hand movement over a period of 4 seconds.  Each trial was followed by a
    short break of at least 1.5 seconds.  A randomized time of up to 1 second
    was added to the break to avoid adaptation

    For the three online feedback sessions four runs with smiley feedback
    were recorded, whereby each run consisted of twenty trials for each type of
    motor imagery.  At the beginning of each trial (second 0) the feedback (a
    gray smiley) was centered on the screen.  At second 2, a short warning beep
    (1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. At
    second 7.5 the screen went blank and a random interval between 1.0 and 2.0
    seconds was added to the trial.

    References
    ----------

    .. [1] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof,
           G. Pfurtscheller. Brain-computer communication: motivation, aim,
           and impact of exploring a virtual apartment. IEEE Transactions on
           Neural Systems and Rehabilitation Engineering 15, 473–482, 2007

    """

    _EVENTS = {
        "left_hand": (1, (3, 7.5)), 
        "right_hand": (2, (3, 7.5)), 
    }

    _CHANNELS = [
        'C3', 'CZ', 'C4'
    ]

    def __init__(self):
        super().__init__(
            code='bnci2014004', 
            subjects=list(range(1, 10)), 
            events=self._EVENTS, 
            channels=self._CHANNELS, 
            srate=250, 
            paradigm='imagery'
        )

    @verbose
    def _get_single_subject_data(self, subject, verbose=False):
        """return data for a single subject"""
        dests = self.data_path(subject)
        sessions = {}
        session_count = 0
        for i_session, session in enumerate(('T', 'E')):
            run_arrays = loadmat(dests[i_session])['data']
            for i_run, run_array in enumerate(run_arrays):
                X = run_array['X'].T
                trial = run_array['trial']
                y = run_array['y']
                stim =  np.zeros((1, X.shape[-1]))

                if y.size > 0:
                    stim[0, trial-1] = y

                data = np.concatenate((X, stim), axis=0)

                ch_names = [ch_name.upper() for ch_name in self._CHANNELS] + ['EOG1', 'EOG2', 'EOG3']
                ch_types = ['eeg']*len(self._CHANNELS) + ['eog']*3
                ch_names = ch_names + ['STI 014']
                ch_types = ch_types + ['stim']
                montage = make_standard_montage('standard_1005')
                # 0.19.2 has no guarantee about case insensitive channel names
                montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
                
                info = create_info(
                    ch_names, self.srate, 
                    ch_types=ch_types, montage=montage
                )

                raw = RawArray(data, info)
                sessions['session_%d' % (session_count)] = {'run_0': raw}
                session_count += 1
        return sessions

    @verbose
    def data_path(self, subject, path=None, force_update=False,
                  update_path=False, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        dests = []
        for run in ['T', 'E']:
            url = '{u:s}004-2014/B{s:02d}{r:s}.mat'.format(u=BNCI_URL, s=subject, r=run)
            dests.append(
                mne_data_path(url, 'bnci', path, force_update, update_path)
            )
        return dests


