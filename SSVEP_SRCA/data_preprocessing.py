# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:45:31 2019
Data preprocessing:
    (1)load data from .cnt file to .mat file
    (2)fitering data
    (3)SRCA optimization
    
Continuously updating...
@author: Brynhildr
"""

#%% load 3rd-part module
import os
import numpy as np
#import mne
import scipy.io as io
#from mne.io import concatenate_raws
#from mne import Epochs, pick_types, find_events
#from mne.baseline import rescale
#from mne.filter import filter_data
import copy
#import srca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fp_growth as fpg
import mcee
#import signal_processing_function as SPF

#%% load data
filepath = r'F:\SSVEP\dataset'

subjectlist = ['wuqiaoyi']

filefolders = []
for subindex in subjectlist:
    filefolder = os.path.join(filepath, subindex)
    filefolders.append(filefolder)

filelist = []
for filefolder in filefolders:
    for file in os.listdir(filefolder):
        filefullpath = os.path.join(filefolder, file)
        filelist.append(filefullpath)

raw_cnts = []
for file in filelist:
    montage = mne.channels.read_montage('standard_1020')
    raw_cnt = mne.io.read_raw_cnt(file, montage=montage,
            eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'],
            preload=True, verbose=False, stim_channel='True')
    # misc=['CB1', 'CB2', 'M1', 'M2'],
    raw_cnts.append(raw_cnt)

raw = concatenate_raws(raw_cnts)

del raw_cnts, file, filefolder, filefolders, filefullpath, filelist
del filepath, subindex, subjectlist

# preprocessing
events = mne.find_events(raw, output='onset')

# drop channels
drop_chans = ['M1', 'M2']

picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False,
                       exclude=drop_chans)
picks_ch_names = [raw.ch_names[i] for i in picks]  # layout picked chans' name

# define labels
event_id = dict(f60p0=1, f60p1=2, f40p0=3, f40p1=4)

#baseline = (-0.2, 0)    # define baseline
tmin, tmax = -2., 2.5    # set the time range
sfreq = 1000

# transform raw object into array
n_stims = int(len(event_id))
n_trials = int(events.shape[0] / n_stims)
n_chans = int(64 - len(drop_chans))
n_times = int((tmax - tmin) * sfreq + 1)
data = np.zeros((n_stims, n_trials, n_chans, n_times))
for i in range(len(event_id)):
    epochs = Epochs(raw, events=events, event_id=i+1, tmin=tmin, picks=picks,
                    tmax=tmax, baseline=None, preload=True)
    data[i,:,:,:] = epochs.get_data()  # (n_trials, n_chans, n_times)
    del epochs
    
del raw, picks, i, n_stims, n_trials, n_chans, n_times
del drop_chans, event_id, events, tmax, tmin

# store raw data
data_path = r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\raw_data.mat'
io.savemat(data_path, {'raw_data':data, 'chan_info':picks_ch_names})

# filtering
data = data[:2,:,:,:]
n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[3]
f_data = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_events):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=50,
                      h_freq=70, n_jobs=4)
del i, data

#%% find & delete bad trials
f_data = np.delete(f_data, [10,33,34,41,42], axis=1)

# store fitered dataw
data_path = r'F:\SSVEP\dataset\preprocessed_data\brynhildr\50_70_bp.mat'
io.savemat(data_path, {'f_data':f_data, 'chan_info':picks_ch_names})

# release RAM
del data_path, f_data, n_chans, n_events, n_times, n_trials, picks_ch_names, sfreq
    
#%% other test
pz = []
po5 = []
po3 = []
poz = []
po4 = []
po6 = []
o1 = []
oz = []
o2 = []

for i in range(10):
    eeg = io.loadmat(r'F:\SSVEP\SRCA data\begin：140ms\OLS\SNR\55_60\srca_%d.mat' %(i))
    exec("model_%d = eeg['model_info'].flatten().tolist()" %(i))
del i, eeg

j = 0
for i in range(10):
    exec("pz.append(model_%d[0+j].tolist())" %(i))
    exec("po5.append(model_%d[1+j].tolist())" %(i))
    exec("po3.append(model_%d[2+j].tolist())" %(i))
    exec("poz.append(model_%d[3+j].tolist())" %(i))
    exec("po4.append(model_%d[4+j].tolist())" %(i))
    exec("po6.append(model_%d[5+j].tolist())" %(i))
    exec("o1.append(model_%d[6+j].tolist())" %(i))
    exec("oz.append(model_%d[7+j].tolist())" %(i))
    exec("o2.append(model_%d[8+j].tolist())" %(i))
del model_0, model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9

#%%
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
ori_data = eeg['f_data'][:,-60:,[45,51,52,53,54,55,58,59,60],2140:2340]*1e6
chan_info = eeg['chan_info'].tolist()

#%%
fig = plt.figure(figsize=(16,12))
gs = GridSpec(4,4, figure=fig)

ax1 = fig.add_subplot(gs[:2,:])
ax1.tick_params(axis='both', labelsize=20)
ax1.set_title('Original signal waveform (OZ)', fontsize=24)
ax1.set_xlabel('Time/ms', fontsize=20)
ax1.set_ylabel('Amplitude/uV', fontsize=20)
ax1.plot(np.mean(ori_data[0,:,7,:], axis=0), label='0 phase')
ax1.plot(np.mean(ori_data[1,:,7,:], axis=0), label='pi phase')
ax1.legend(loc='upper left', fontsize=16)

ax2 = fig.add_subplot(gs[2:,:])
ax2.tick_params(axis='both', labelsize=20)
ax2.set_title('Fisher score optimized signal waveform (OZ)', fontsize=24)
ax2.set_xlabel('Time/ms', fontsize=20)
ax2.set_ylabel('Amplitude/uV', fontsize=20)
ax2.plot(np.mean(srca_data[0,:,7,:], axis=0), label='0 phase')
ax2.plot(np.mean(srca_data[1,:,7,:], axis=0), label='pi phase')
ax2.legend(loc='upper left', fontsize=16)

plt.show()

#%% FP-Growth
if __name__ == '__main__':
    '''
    Call function 'find_frequent_itemsets()' to form frequent items
    '''
    frequent_itemsets = fpg.find_frequent_itemsets(oz, minimum_support=5,
                                                   include_support=True)
    #print(type(frequent_itemsets))
    result = []
    # save results from generator into list
    for itemset, support in frequent_itemsets:  
        result.append((itemset, support))
    # ranking
    result = sorted(result, key=lambda i: i[0])
    print('FP-Growth complete!')

#%%
compressed_result = []
number = []
for i in range(len(result)):
    if len(result[i][0]) > 3:
        compressed_result.append(result[i][0])
        number.append(result[i][1])
del i

#%%
delta = np.zeros((4,10,18))
summ = np.zeros((4,10,18))
for i in range(4):
    for j in range(10):
        eeg = io.loadmat(r'F:\SSVEP\ols\ols+snr\real1_%d\mcee_%d.mat' %(i+1,j))
        parameter_ols = eeg['parameter'].flatten().tolist()
        para_ols = []
        for k in range(len(parameter_ols)):
            para_ols.append(np.max(parameter_ols[k]))   
        eeg = io.loadmat(r'F:\SSVEP\ridge\ridge+snr\real1_%d\mcee_%d.mat' %(i+1,j))
        parameter_ri = eeg['parameter'].flatten().tolist()
        para_ri = []
        for l in range(len(parameter_ri)):
            para_ri.append(np.max(parameter_ri[l]))
        para_ols = np.array(para_ols)
        para_ri = np.array(para_ri)
        delta[i,j,:] = para_ri - para_ols
        summ[i,j,:] = para_ri + para_ols
ratio = (np.mean((delta+summ)/(summ-delta))-1)*100

#%% real te tr
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS', 'Ridge']
methodList = ['FS']
trainNum = [55, 40, 30, 25]

n_events = 2
n_trials = 115
n_chans = len(tar_chans)
n_test = 60
n_times = 2140

for reg in range(len(regressionList)):
    regression = regressionList[reg]
    for met in range(len(methodList)):
        method = methodList[met]
        for nfile in range(len(trainNum)):
            ns = trainNum[nfile]
            for nt in range(5):
                model_info = []
                snr_alteration = []
                for ntc in range(len(tar_chans)):
                    target_channel = tar_chans[ntc]
                    sfreq = 1000
                    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
                    temp = eeg['f_data'][:,:,:,1000:3140] * 1e6
                    f_data = np.zeros_like(temp)
                    f_data[0,:,:,:] = temp[0,:,:,:]
                    f_data[1,:,:,:] = eeg['f_data'][1,:,:,1010:3150] * 1e6
                    w = f_data[:,:ns,:,:1000]
                    signal_data = f_data[:,:ns,:,1140:int(1300+nt*100)]
                    chans = eeg['chan_info'].tolist() 
                    del eeg, temp
                    w_o = w[:,:,chans.index(target_channel),:]
                    w_temp = copy.deepcopy(w)
                    w_i = np.delete(w_temp, chans.index(target_channel), axis=2)
                    del w_temp
                    sig_o = signal_data[:,:,chans.index(target_channel),:]
                    sig_temp = copy.deepcopy(signal_data)
                    sig_i = np.delete(sig_temp, chans.index(target_channel), axis=2)
                    del sig_temp, signal_data
                    srca_chans = copy.deepcopy(chans)
                    del srca_chans[chans.index(target_channel)]
                    msnr = np.mean(mcee.fisher_score(sig_o)) 
                    model_chans, para_change = mcee.stepwise_SRCA_fs(srca_chans,
                            msnr, w, w_o, sig_i, sig_o, regression)
                    snr_alteration.append(para_change)
                    model_info.append(model_chans)
                data_path = r'F:\SSVEP\SRCA data\begin：200ms-0.4pi\%s\%s\%d_60\srca_%d.mat' %(regression, method, ns, nt)             
                io.savemat(data_path, {'model_info': model_info, 'parameter': snr_alteration})

#%% e-trca for origin data
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
trainNum = [55, 40, 30, 25]
acc_srca_ori = np.zeros((5, 10))
for nt in range(10):
    eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
    temp = eeg['f_data'][:,-60:,[45,51,52,53,54,55,58,59,60],2200:int(2300+nt*100)]*1e6
    data = np.zeros_like(temp)
    data[0,:,:,:] = temp[0,:,:,:]
    data[1,:,:,:] = np.swapaxes(eeg['f_data'][1,-60:,[45,51,52,53,54,55,58,59,60],2200:int(2300+nt*100)]*1e6, 0, 1)
    del eeg, temp
    acc = []
    N = 5
    print('running ensemble TRCA...')
    for cv in range(N):
        a = int(cv * (data.shape[1]/N))
        tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
        te_data = copy.deepcopy(data)
        te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
        acc_temp = mcee.e_trca(te_data, tr_data)
        acc.append(np.sum(acc_temp))
        del acc_temp
        print(str(cv+1) + 'th cv complete')
    acc = np.array(acc)/(tr_data.shape[1]*2)
    acc_srca_ori[:, nt] = acc
    del acc

#%% trca for origin data: new method
trainNum = [55, 40, 30, 25]
acc_srca_ori = np.zeros((4,5))
for nfile in range(len(trainNum)):
    for nt in range(5):
        acc = []
        print('data length: %d00ms' %(nt+1))
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
        tr_data = eeg['f_data'][:,:trainNum[nfile],[45,51,52,53,54,55,58,59,60],2140:int(2240+nt*100)]*1e6
        te_data = eeg['f_data'][:,-60:,[45,51,52,53,54,55,58,59,60],2140:int(2240+nt*100)]*1e6
        acc_temp = mcee.pure_trca(tr_data, te_data)
        acc.append(np.sum(acc_temp))
        del acc_temp
        acc_srca_ori[nfile,nt] = np.array(acc)/(te_data.shape[1]*2)
        del acc
    
#%% ensemble trca
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS']
#regressionList = ['OLS', 'Ridge']
#methodList = ['SNR', 'Corr']
methodList = ['SNR']
trainNum = [55]
#trainNum = [55, 40, 30, 25]

acc_srca_tr = np.zeros((len(regressionList), len(methodList), len(trainNum), 5))
for reg in range(len(regressionList)):
    regression = regressionList[reg]
    for met in range(len(methodList)):
        method = methodList[met]
        for nfile in range(len(trainNum)):
            ns = trainNum[nfile]
            for nt in range(5):
                # extract model info used in SRCA
                eeg = io.loadmat(r'F:\SSVEP\SRCA data\begin：140ms-0.4pi\%s\%s\%d_60\srca_%d.mat'
                                     %(regression, method, ns, nt))
                model = eeg['model_info'].flatten().tolist()
                model_chans = []
                for i in range(18):
                    model_chans.append(model[i].tolist())
                print('Data length:' + str((nt+1)*100) + 'ms')
                del model, eeg       
                # extract origin data
                eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
                temp_tr = eeg['f_data'][:, :ns, :, 1000:int(2240+nt*100)]*1e6
                tr_data = np.zeros_like(temp_tr)
                tr_data[0,:,:,:] = temp_tr[0,:,:,:]
                tr_data[1,:,:,:] = eeg['f_data'][1, :ns, :, 1000:int(2240+nt*100)]*1e6
                temp_te = eeg['f_data'][:, -60:, :, 1000:int(2240+nt*100)]*1e6
                te_data = np.zeros_like(temp_te)
                te_data[0,:,:,:] = temp_te[0,:,:,:]
                te_data[1,:,:,:] = eeg['f_data'][1, -60:, :, 1000:int(2240+nt*100)]*1e6
                chans = eeg['chan_info'].tolist()
                del eeg, temp_tr, temp_te
                # cross validation
                acc = []
                acc_temp = mcee.srca_trca(train_data=tr_data, test_data=te_data,
                    tar_chans=tar_chans, model_chans=model_chans, chans=chans,
                    regression=regression, sp=1140)
                acc.append(np.sum(acc_temp))
                del acc_temp
                acc = np.array(acc)/(te_data.shape[1]*2)
                acc_srca_tr[reg, met, nfile, nt] = acc
                del acc
                
#%%
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
regressionList = ['OLS']
#regressionList = ['OLS', 'Ridge']
methodList = ['SNR', 'Corr']
#methodList = ['SNR']
#trainNum = [55]
trainNum = [55, 40, 30, 25]
acc_srca_te = np.zeros((len(regressionList), len(methodList), len(trainNum), 5, 5))
for reg in range(len(regressionList)):
    regression = regressionList[reg]
    for met in range(len(methodList)):
        method = methodList[met]
        for nfile in range(len(trainNum)):
            ns = trainNum[nfile]
            for nt in range(5):
                # extract model info used in SRCA
                eeg = io.loadmat(r'F:\SSVEP\SRCA data\begin：200ms\%s\%s\%d_60\srca_%d.mat'
                                     %(regression, method, ns, nt))
                model = eeg['model_info'].flatten().tolist()
                model_chans = []
                for i in range(18):
                    model_chans.append(model[i].tolist())
                print('Data length:' + str((nt+1)*100) + 'ms')
                del model, eeg       
                # extract origin data
                eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
                temp = eeg['f_data'][:,-60:,:,1000:int(2300+nt*100)]*1e6
                data = np.zeros_like(temp)
                data[0,:,:,:] = temp[0,:,:,:]
                data[1,:,:,:] = eeg['f_data'][1,-60:,:,1000:int(2300+nt*100)]*1e6
                chans = eeg['chan_info'].tolist()
                del eeg, temp
                # cross validation
                acc = []
                N = 5
                print('running TRCA program...')
                for cv in range(N):
                    a = int(cv * (data.shape[1]/N))
                    tr_data = data[:,a:a+int(data.shape[1]/N),:,:]
                    te_data = copy.deepcopy(data)
                    te_data = np.delete(te_data, [a+i for i in range(tr_data.shape[1])], axis=1)
                    acc_temp = mcee.srca_trca(train_data=te_data, test_data=tr_data,
                        tar_chans=tar_chans, model_chans=model_chans, chans=chans,
                        regression=regression, sp=1200)
                    acc.append(np.sum(acc_temp))
                    del acc_temp
                    print(str(cv+1) + 'th cv complete')
                acc = np.array(acc)/(tr_data.shape[1]*2)
                acc_srca_te[reg, met, nfile, :, nt] = acc
                del acc
                
#%%
data_path = r'F:\SSVEP\SRCA data\begin：140ms\trca_result.mat'
io.savemat(data_path, {'tr<te':acc_srca_tr, 'te<tr':acc_srca_te})

#%%
regressionList = ['OLS']
trainNum = [55, 40, 30, 25]
for reg in range(len(regressionList)):
    for nfile in range(len(trainNum)):
        for nt in range(10):
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
            train_data = eeg['f_data'][:, -60:, :, 1000:2300+nt*100]*1e6
            chans = eeg['chan_info'].tolist()
            del eeg
            tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
            eeg = io.loadmat(r'F:\SSVEP\SRCA data\begin：200ms\%s\FS\%d_60\srca_%d'
                             %(regressionList[reg], trainNum[nfile], nt))
            model = eeg['model_info'].flatten().tolist()
            model_chans = []
            for i in range(len(model)):
                model_chans.append(model[i].tolist())
            del eeg, i, model
            n_events = train_data.shape[0]
            n_trains = train_data.shape[1]
            n_chans = len(tar_chans)
            n_times = train_data.shape[-1] - 1200
            model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
            for ntc in range(len(tar_chans)):
                target_channel = tar_chans[ntc]
                model_chan = model_chans[ntc]
                w_i = np.zeros((n_events, n_trains, len(model_chan), 1000))
                sig_i = np.zeros((n_events, n_trains, len(model_chan), n_times))
                for nc in range(len(model_chan)):
                    w_i[:, :, nc, :] = train_data[:, :, chans.index(model_chan[nc]), :1000]
                    sig_i[:, :, nc, :] = train_data[:, :, chans.index(model_chan[nc]), 1200:]
                del nc
                w_o = train_data[:, :, chans.index(target_channel), :1000]
                sig_o = train_data[:, :, chans.index(target_channel), 1200:]
                w_i = np.swapaxes(w_i, 1, 2)
                sig_i = np.swapaxes(sig_i, 1, 2)
                for ne in range(n_events):
                    w_ex_s = mcee.mlr(w_i[ne, :, :, :], w_o[ne, :, :],
                          sig_i[ne, :, :, :], sig_o[ne, :, :], regressionList[reg])
                    model_sig[ne, :, ntc, :] = w_ex_s
                del ne
            del ntc, model_chan, w_i, w_o, sig_i, sig_o, w_ex_s
            data_label = [0 for i in range(60)]
            data_label += [1 for i in range(60)]
            srca_data = np.zeros((120, 9, n_times))
            srca_data[:60, :, :] = model_sig[0,:,:,:]
            srca_data[60:, :, :] = model_sig[1,:,:,:]
            srca_data = np.swapaxes(srca_data, 0, -1)
            del model_sig, train_data
            data_path = r'F:\SSVEP\DCPM_Pre\begin：200ms\%s\%d_60\t%d.mat' %(regressionList[reg],
                    trainNum[nfile], nt)
            io.savemat(data_path, {'srca_data':srca_data, 'label':data_label})
            print('Method:{}--Data length:{}ms--{} samples complete!'.format(regressionList[reg],
                    100*(nt+1), trainNum[nfile]))

#%%
n_events = train_data.shape[0]
n_trains = train_data.shape[1]
n_chans = train_data.shape[2]
n_times = train_data.shape[-1]
    
template = np.mean(train_data, axis=1)
#%%
q = np.zeros((n_events, n_chans, n_chans))
for x in range(n_events):
    temp = np.zeros((n_chans, int(n_trains*n_times)))
    for y in range(n_chans):
        temp[y,:] = train_data[x,:,y,:].flatten()
    del y
    q[x,:,:] = np.cov(temp)
    del temp
del x
#%%
s = np.zeros((n_events, n_chans, n_chans))
for v in range(n_events):  # v for events
    for w in range(n_chans):  # w for channels (j1)
        for x in range(n_chans):  # x for channels (j2)
            cov = []
            for y in range(n_trains):  # y for trials (h1)
                temp = np.zeros((2, n_times))
                temp[0,:] = train_data[v,y,w,:]
                for z in range(n_trains):  # z for trials (h2)
                    if z != y:  # h1 != h2
                        temp[1,:] = train_data[v,z,x,:]
                        cov.append(np.sum(np.tril(np.cov(temp),-1)))
                    else:
                        continue
                del z, temp
            del y
            s[v,w,x] = np.sum(cov)
            del cov
        del x
    del w
del v
#%% 
w = np.zeros((n_events, n_chans))
for z in range(n_events):
    qs = np.mat(q[z,:,:]).I * np.mat(s[z,:,:])
    e_value, e_vector = np.linalg.eig(qs)
    w_index = np.max(np.where(e_value == np.max(e_value)))
    w[z,:] = e_vector[:,w_index].T
    del w_index
del z
#%%
r = np.zeros((n_events, test_data.shape[1], n_events))
for x in range(n_events):
    for y in range(test_data.shape[1]):
        for z in range(n_events):
            temp_test = (np.mat(w) * np.mat(test_data[x,y,:,:])).flatten()
            temp_template = (np.mat(w) * np.mat(template[z,:,:])).flatten()
            r[x,y,z] = np.sum(np.tril(np.corrcoef(temp_test,temp_template), -1))
        del z, temp_test, temp_template
    del y
del x
#%% 
acc = []
for x in range(r.shape[0]):
    for y in range(r.shape[1]):
        if np.max(np.where(r[x,y,:] == np.max(r[x,y,:]))) == x:
            acc.append(1)
            
#%%
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\50_70_bp.mat')
for nt in range(10):
    data = eeg['f_data'][:,-60:,[45,51,52,53,54,55,58,59,60],2140:2240+nt*100]*1e6
    temp = data[:,:,:,:(nt+1)*100]
    srca_data = np.zeros((120,9,(nt+1)*100))
    data_path = r'F:\SSVEP\DCPM_Pre\origin：0.4pi\t%d.mat' %(nt)
    srca_data[:60,:,:] = temp[0,:,:,:]
    srca_data[60:,:,:] = data[1,:,:,:(nt+1)*100+10]
    label = [0 for i in range(60)]
    label += [1 for i in range(60)]
    srca_data = np.swapaxes(srca_data, 0, -1)
    io.savemat(data_path, {'srca_data':srca_data, 'label':label})
    
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15,10))
gs = GridSpec(2,2,figure=fig)

ax1 = fig.add_subplot(gs[:,:])
ax1.set_title('Origin Signal', fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(np.mean(data[0,:,7,:], axis=0), label='0 phase')
ax1.plot(np.mean(data[1,:,7,:], axis=0), label='pi phase')
ax1.set_xlabel('Time/ms', fontsize=22)
ax1.set_ylabel('Amplitude/μV', fontsize=22)
ax1.vlines(140, -1, 1, color='black', linestyle='dashed', label='140ms')
ax1.legend(loc='upper left', fontsize=20)

fig.tight_layout()
plt.show()
plt.savefig(r'C:\Users\brynh\Desktop\fuck.png', dpi=600)