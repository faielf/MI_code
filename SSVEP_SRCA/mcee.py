# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:07:56 2019

1. Basic operating functions:
    (1)Multi_linear regression: mlr
    (2)SNR in time domain: snr_time
    
2. Three kinds of recursive algorithm to choose channels for optimization
    (1)Backward: backward_MCEE
    (2)Forward: forward_MCEE
    (3)Stepwise: stepwise_MCEE

3. Target identification
    (1)TRCA
    (2)Correlation dectection
    (3)Ensembled-TRCA (future version)
    (4)FB-CCA (future version)
    (5)CCA (future version)
    (6)PS

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
from sklearn import linear_model
import copy
import time

#%% Basic operating function
# multi-linear regression
def mlr(model_input, model_target, data_input, data_target, regression):
    '''
    model_input: (n_chans, n_trials, n_times)
    model_target: (n_trials, n_times)
    data_input: (n_chans, n_trials, n_times)
    data_target: (n_trials, n_times)
    '''
    if model_input.ndim == 3:
        # estimate signal: (n_trials, n_times)
        estimate = np.zeros((data_input.shape[1], data_input.shape[2]))
        for i in range(model_input.shape[1]):    # i for trials
            # basic operating unit: (n_chans, n_times).T, (1, n_times).T
            if regression == 'OLS':
                L = linear_model.LinearRegression().fit(model_input[:,i,:].T, model_target[i,:].T)
            elif regression == 'Ridge':
                L = linear_model.Ridge().fit(model_input[:,i,:].T, model_target[i,:].T)
            elif regression == 'Lasso':
                L = linear_model.Lasso().fit(model_input[:,i,:].T, model_target[i,:].T)
            elif regression == 'ElasticNet':
                L = linear_model.ElasticNet().fit(model_input[:,i,:].T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = (np.mat(RC) * np.mat(data_input[:,i,:])).A + RI           
    elif model_input.ndim == 2:      # avoid reshape error
        RI = np.zeros((model_input.shape[0]))
        estimate = np.zeros((model_input.shape[0], data_input.shape[1]))
        RC = np.zeros((model_input.shape[0]))
        for i in range(model_input.shape[0]):
            if regression == 'OLS':
                L = linear_model.LinearRegression().fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            elif regression == 'Ridge':
                L = linear_model.Ridge().fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            elif regression == 'Lasso':
                L = linear_model.Lasso().fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            elif regression == 'ElasticNet':
                L = linear_model.ElasticNet().fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = RC * data_input[i,:] + RI
    # extract SSVEP from raw data
    extract = data_target - estimate
    return extract

# compute time-domain snr
def snr_time(data):
    '''
    data:(n_trials, n_times)
    '''
    snr = np.zeros((data.shape[1]))             # (n_times)
    ex = np.mat(np.mean(data, axis=0))          # one-channel data: (1, n_times)
    temp = np.mat(np.ones((1, data.shape[0])))  # (1, n_trials)
    minus = (temp.T * ex).A                     # (n_trials, n_times)
    ex = (ex.A) ** 2                            # signal's power
    var = np.mean((data - minus)**2, axis=0)    # noise's power (avg)
    snr = ex/var
    return snr

def pearson_corr(data):
    '''
    data | (n_trials, n_times): input 3-D data array
    corr | pearson correlation coefficients sequence
    mcorr | mean of sequence
    '''
    template = np.mean(data, axis=0)
    n_trials = data.shape[0]
    corr = np.zeros((n_trials))
    for i in range(n_trials):
        corr[i] = np.sum(np.tril(np.corrcoef(template, data[i,:]),-1))
    return corr

def fisher_score(data):
    '''
    data: (n_events, n_trails, n_times)
    '''
    # initialization
    sampleNum = data.shape[1]    # n_trials
    featureNum = data.shape[-1]  # n_times
    groupNum = data.shape[0]     # n_events
    miu = np.mean(data, axis=1)  # (n_events, n_times)
    all_miu = np.mean(miu, axis=0)
    # inter-class divergence
    ite_d = np.sum(sampleNum * (miu - all_miu)**2, axis=0)
    # intra-class divergence
    itr_d= np.zeros((groupNum, featureNum))
    for i in range(groupNum):
        for j in range(featureNum):
            itr_d[i,j] = np.sum((data[i,:,j] - miu[i,j])**2)
    # fisher score
    fs = (ite_d) / np.sum(sampleNum * itr_d, axis=0)
    return fs

#%% Backward MCEE
def backward_MCEE(chans, msnr, w, w_target, signal_data, data_target):
    '''
    Backward recursive algorithm to achieve MCEE
        (1)take all possible channels to form a model
        (2)delete one element each time respectively and keep the best choice
        (3)repeat the lase process until there will be no better choice
            i.e. the convergence point of the recursive algorithm
    Parameters:
        chans: list of channels; the list order corresponds to the data array's
        msnr: float; the mean of original signal's SNR in time domain(0-500ms)
        w: background part input data array (n_trials, n_chans, n_times)
        w_target: background part target data array (n_trials, n_times)
        signal_data: signal part input data array (n_trials, n_chans, n_times)
        data_target: signal part target data array (n_trials, n_times)
    Returns:
        model_chans: list of channels which should be used in MCEE
        snr_change: list of SNR's alteration
    '''
    # initialize variables
    print('Running Backward EE...')
    start = time.clock()  
    j = 0 
    compare_snr = np.zeros((len(chans)))
    delete_chans = []
    snr_change = []
    # begin loop
    active = True
    while active:
        if len(chans) > 1:
            compare_snr = np.zeros((len(chans)))
            mtemp_snr = np.zeros((len(chans)))
            # delete 1 channel respectively and compare the snr with original one
            for i in range(len(chans)):
                # initialization
                temp_chans = copy.deepcopy(chans)
                temp_data = copy.deepcopy(signal_data)
                temp_w = copy.deepcopy(w)
                # delete one channel
                del temp_chans[i]
                temp_data = np.delete(temp_data, i, axis=1)
                temp_w = np.delete(temp_w, i, axis=1)
                # compute snr
                temp_extract = mlr(temp_w, w_target, temp_data, data_target)
                temp_snr = snr_time(temp_extract)
                mtemp_snr[i] = np.mean(temp_snr)
                compare_snr[i] = mtemp_snr[i] - msnr
            # keep the channels which can improve snr forever
            chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
            delete_chans.append(chans.pop(chan_index))
            snr_change.append(np.max(compare_snr))
            # refresh data
            signal_data = np.delete(signal_data, chan_index, axis=1)
            w = np.delete(w, chan_index, axis=1)
            # significant loop mark
            j += 1 
            print('Complete ' + str(j) + 'th loop')
        # Backward EE complete
        else:
            end = time.clock()
            print('Backward EE complete!')
            print('Recursive running time: ' + str(end - start) + 's')
            active = False      
    model_chans = chans + delete_chans[-2:]
    return model_chans, snr_change

#%% Forward MCEE
def forward_MCEE(chans, msnr, w, w_target, signal_data, data_target):
    '''
    Forward recursive algorithm to achieve MCEE
    Contrary to Backward process:
        (1)this time form an empty set
        (2)add one channel each time respectively and keep the best choice
        (3)repeat the last process until there will be no better choice
            i.e. the convergence point of the recursive algorithm
    Parameters:
        chans: list of channels; the list order corresponds to the data array's
        msnr: float; the mean of original signal's SNR in time domain(0-500ms)
        w: background part input data array (n_trials, n_chans, n_times)
        w_target: background part target data array (n_trials, n_times)
        signal_data: signal part input data array (n_trials, n_chans, n_times)
        data_target: signal part target data array (n_trials, n_times)
    Returns:
        model_chans: list of channels which should be used in MCEE
        snr_change: list of SNR's alteration
    '''
    # initialize variables
    print('Running Forward EE...')
    start = time.clock()   
    j = 1   
    compare_snr = np.zeros((len(chans)))
    max_loop = len(chans)
    remain_chans = []
    snr_change = []
    temp_snr = []
    core_data = []
    core_w = []
    # begin loop
    active = True
    while active and len(chans) <= max_loop:
        # initialization
        compare_snr = np.zeros((len(chans)))
        mtemp_snr = np.zeros((len(chans)))
        # add 1 channel respectively and compare the snr with original one
        for i in range(len(chans)):
            # avoid reshape error in multi-dimension array
            if j == 1:
                temp_w = w[:,i,:]
                temp_data = signal_data[:,i,:]
            else:
                temp_w = np.zeros((j, w.shape[0], w.shape[2]))
                temp_w[:j-1, :, :] = core_w
                temp_w[j-1, :, :] = w[:,i,:]
        
                temp_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
                temp_data[:j-1, :, :] = core_data
                temp_data[j-1, :, :] = signal_data[:,i,:]
            # multi-linear regression & snr computation
            temp_extract = mlr(temp_w, w_target, temp_data, data_target)
            temp_snr = snr_time(temp_extract)
            # find the best choice in this turn
            mtemp_snr[i] = np.mean(temp_snr)
            compare_snr[i] = mtemp_snr[i] - msnr
        # keep the channels which can improve snr most
        chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
        remain_chans.append(chans.pop(chan_index))
        snr_change.append(np.max(compare_snr))
        # avoid reshape error at the beginning of Forward EE
        if j == 1:
            core_w = w[:, chan_index, :]
            core_data = signal_data[:, chan_index, :]
            # refresh data
            signal_data = np.delete(signal_data, chan_index, axis=1)
            w = np.delete(w ,chan_index, axis=1)
            # significant loop mark
            print('Complete ' + str(j) + 'th loop')
            j += 1
        else:
            temp_core_w = np.zeros((j, w.shape[0], w.shape[2]))
            temp_core_w[:j-1, :, :] = core_w
            temp_core_w[j-1, :, :] = w[:, chan_index, :]
            core_w = copy.deepcopy(temp_core_w)
            del temp_core_w
            
            temp_core_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
            temp_core_data[:j-1, :, :] = core_data
            temp_core_data[j-1, :, :] = signal_data[:, chan_index, :]
            core_data = copy.deepcopy(temp_core_data)
            del temp_core_data
            # add judge condition to stop program while achieving the target
            if snr_change[j-1] < np.max(snr_change):
                end = time.clock()
                print('Forward EE complete!')
                print('Recursive running time: ' + str(end - start) + 's')
                active = False
            else:
                # refresh data
                signal_data = np.delete(signal_data, chan_index, axis=1)
                w = np.delete(w ,chan_index, axis=1)
                # significant loop mark
                print('Complete ' + str(j) + 'th loop')
                j += 1
        
    remain_chans = remain_chans[:len(remain_chans)-1]
    return remain_chans, snr_change

#%% Stepwise MCEE
def stepwise_MCEE(chans, msnr, w, w_target, signal_data, data_target, method,
                  regression):
    '''
    Stepward recursive algorithm to achieve MCEE
    The combination of Forward and Backward process:
        (1)this time form an empty set; 
        (2)add one channel respectively and pick the best one; 
        (3)add one channel respectively and delete one respectively (except the just-added one)
            keep the best choice;
        (4)repeat those process until there will be no better choice
            i.e. the convergence point of the recursive algorithm
    Parameters:
        chans: list of channels; the list order corresponds to the data array's
        msnr: float; the mean of original signal's SNR in time domain(0-500ms)
        w: background part input data array (n_trials, n_chans, n_times)
        w_target: background part target data array (n_trials, n_times)
        signal_data: signal part input data array (n_trials, n_chans, n_times)
        data_target: signal part target data array (n_trials, n_times)
    Returns:
        model_chans: list of channels which should be used in MCEE
        snr_change: list of SNR's alteration
    '''
    # initialize variables
    print('Running Stepwise MCEE...')
    start = time.clock()   
    j = 1   
    compare_snr = np.zeros((len(chans)))
    max_loop = len(chans)   
    remain_chans = []
    snr_change = []
    temp_snr = []
    core_data = []
    core_w = []
    # begin loop
    active = True
    while active and len(chans) <= max_loop:
        # initialization
        compare_snr = np.zeros((len(chans)))
        mtemp_snr = np.zeros((len(chans)))
        # add 1 channel respectively & compare the snr (Forward EE)
        for i in range(len(chans)):
            # avoid reshape error in multi-dimension array
            if j == 1:
                temp_w = w[:,i,:]
                temp_data = signal_data[:,i,:]
            else:
                temp_w = np.zeros((j, w.shape[0], w.shape[2]))
                temp_w[:j-1, :, :] = core_w
                temp_w[j-1, :, :] = w[:,i,:]                
                temp_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
                temp_data[:j-1, :, :] = core_data
                temp_data[j-1, :, :] = signal_data[:,i,:]
            # multi-linear regression & snr computation
            temp_extract = mlr(model_input=temp_w, model_target=w_target,
                               data_input=temp_data, data_target=data_target,
                               regression=regression)
            del temp_w, temp_data
            if method == 'SNR':
                temp_snr = snr_time(temp_extract)
            elif method == 'Corr':
                temp_snr = pearson_corr(temp_extract)
            # compare the snr with original one
            mtemp_snr[i] = np.mean(temp_snr)
            compare_snr[i] = mtemp_snr[i] - msnr
        # keep the channels which can improve snr most
        chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
        remain_chans.append(chans.pop(chan_index))
        snr_change.append(np.max(compare_snr))
        del temp_extract, compare_snr, mtemp_snr, temp_snr      
        # avoid reshape error at the beginning of Forward EE while refreshing data
        if j == 1: 
            core_w = w[:, chan_index, :]
            core_data = signal_data[:, chan_index, :]
            # refresh data
            signal_data = np.delete(signal_data, chan_index, axis=1)
            w = np.delete(w ,chan_index, axis=1)
            # significant loop mark
            print('Complete ' + str(j) + 'th loop')        
        # begin stepwise part (delete & add) 
        if j == 2:  
            # save new data
            temp_core_w = np.zeros((j, w.shape[0], w.shape[2]))
            temp_core_w[0, :, :] = core_w
            temp_core_w[1, :, :] = w[:, chan_index, :]
            core_w = copy.deepcopy(temp_core_w)
            del temp_core_w
            temp_core_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
            temp_core_data[0, :, :] = core_data
            temp_core_data[1, :, :] = signal_data[:, chan_index, :]
            core_data = copy.deepcopy(temp_core_data)
            del temp_core_data
            # add judge condition to stop program while achieving the target
            if snr_change[-1] < np.max(snr_change):
                print('Stepwise EE complete!')
                end = time.clock()
                print('Recursive running time: ' + str(end - start) + 's')
                # if this judgement is not passed, then there's no need to continue
                active = False
            # delete 1st channel, then add a new one
            else:
                # refresh data
                signal_data = np.delete(signal_data, chan_index, axis=1)
                w = np.delete(w, chan_index, axis=1)
                # initialization
                temp_1_chans = copy.deepcopy(remain_chans)
                temp_1_data = copy.deepcopy(core_data)
                temp_1_w = copy.deepcopy(core_w)
                # delete 1st channel
                del temp_1_chans[0]
                temp_1_data = np.delete(temp_1_data, 0, axis=0)
                temp_1_w = np.delete(temp_1_w, 0, axis=0)
                # add one channel
                temp_2_compare_snr = np.zeros((signal_data.shape[1]))
                for k in range(signal_data.shape[1]):
                    temp_2_w = np.zeros((2, w.shape[0], w.shape[2]))
                    temp_2_w[0, :, :] = temp_1_w
                    temp_2_w[1, :, :] = w[:, k, :]
                    temp_2_data = np.zeros((2, signal_data.shape[0], signal_data.shape[2]))
                    temp_2_data[0, :, :] = temp_1_data
                    temp_2_data[1, :, :] = signal_data[:, k, :]
                    # mlr & compute snr
                    temp_2_extract = mlr(model_input=temp_2_w, model_target=w_target,
                                         data_input=temp_2_data, data_target=data_target,
                                         regression=regression)
                    if method == 'SNR':
                        temp_2_snr = snr_time(temp_2_extract)
                    elif method == 'Corr':
                        temp_2_snr = pearson_corr(temp_2_extract)
                    mtemp_2_snr = np.mean(temp_2_snr)
                    temp_2_compare_snr[k] = mtemp_2_snr - msnr
                # keep the best choice
                temp_2_chan_index = np.max(np.where(temp_2_compare_snr == np.max(temp_2_compare_snr)))
                # judge if there's any improvement
                if temp_2_compare_snr[temp_2_chan_index] > snr_change[-1]:  # has improvement
                    # refresh data
                    chan_index = temp_2_chan_index
                    remain_chans.append(chans.pop(chan_index))
                    snr_change.append(temp_2_compare_snr[temp_2_chan_index])
                    # delete useless data & add new data
                    core_w = np.delete(core_w, 0, axis=0)
                    temp_2_core_w = np.zeros((2, w.shape[0], w.shape[2]))
                    temp_2_core_w[0, :, :] = core_w
                    temp_2_core_w[1, :, :] = w[:, chan_index, :]
                    core_w = copy.deepcopy(temp_2_core_w)
                    del temp_2_core_w
                    core_data = np.delete(core_data, 0, axis=0)
                    temp_2_core_data = np.zeros((2, signal_data.shape[0], signal_data.shape[2]))
                    temp_2_core_data[0, :, :] = core_data
                    temp_2_core_data[1, :, :] = signal_data[:, chan_index, :]
                    core_data = copy.deepcopy(temp_2_core_data)
                    del temp_2_core_data
                    signal_data = np.delete(signal_data, chan_index, axis=1)
                    w = np.delete(w, chan_index, axis=1)
                    # release RAM
                    del remain_chans[0]
                    del temp_2_chan_index, temp_2_extract, temp_2_snr
                    del mtemp_2_snr, temp_2_compare_snr, temp_2_w, temp_2_data
                    del temp_1_chans, temp_1_data, temp_1_w
                    # significant loop mark
                    print('Complete ' + str(j) + 'th loop')               
                else:  # no improvement
                    # release RAM
                    del temp_1_chans, temp_1_data, temp_1_w
                    # reset
                    print("Already best in 2 channels' contidion!")
        # now we have at least 3 elements in remain_chans,
        # delete one channel, then add a new one
        if j > 2:
            # save data
            temp_core_w = np.zeros((j, w.shape[0], w.shape[2]))
            temp_core_w[:j-1, :, :] = core_w
            temp_core_w[j-1, :, :] = w[:, chan_index, :]
            core_w = copy.deepcopy(temp_core_w)
            del temp_core_w
            temp_core_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
            temp_core_data[:j-1, :, :] = core_data
            temp_core_data[j-1, :, :] = signal_data[:, chan_index, :]
            core_data = copy.deepcopy(temp_core_data)
            del temp_core_data
            # add judge condition to stop program while achieving the target
            if snr_change[-1] < np.max(snr_change):
                print('Stepwise EE complete!')
                end = time.clock()
                print('Recursive running time: ' + str(end - start) + 's')
                # if this judge is not passed, then there's no need to continue
                active = False
            # now the last snr_change is still the largest in the total sequence
            else:
                # refresh data
                signal_data = np.delete(signal_data, chan_index, axis=1)
                w = np.delete(w, chan_index, axis=1)
                # initialization (make copies)
                temp_3_chans = copy.deepcopy(remain_chans)
                temp_3_compare_snr = np.zeros((len(temp_3_chans)-1))
                temp_3_chan_index = []
                # delete one channel except the latest one
                for l in range(len(temp_3_chans)-1):
                    # initialization (make copies)
                    temp_4_chans = copy.deepcopy(remain_chans)
                    temp_4_data = copy.deepcopy(core_data)
                    temp_4_w = copy.deepcopy(core_w)
                    # delete one channel
                    del temp_4_chans[l]
                    temp_4_data = np.delete(temp_4_data, l, axis=0)
                    temp_4_w = np.delete(temp_4_w, l, axis=0)
                    # add one channel
                    temp_4_compare_snr = np.zeros((signal_data.shape[1]))
                    for m in range(signal_data.shape[1]):
                        temp_5_w = np.zeros((j, w.shape[0], w.shape[2]))
                        temp_5_w[:j-1, :, :] = temp_4_w
                        temp_5_w[j-1, :, :] = w[:, m, :]               
                        temp_5_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
                        temp_5_data[:j-1, :, :] = temp_4_data
                        temp_5_data[j-1, :, :] = signal_data[:, m, :]
                        # mlr & compute snr
                        temp_5_extract = mlr(model_input=temp_5_w, model_target=w_target,
                                             data_input=temp_5_data, data_target=data_target,
                                             regression=regression)
                        if method == 'SNR':
                            temp_5_snr = snr_time(temp_5_extract)
                        elif method == 'Corr':
                            temp_5_snr = pearson_corr(temp_5_extract)
                        mtemp_5_snr = np.mean(temp_5_snr)
                        temp_4_compare_snr[m] = mtemp_5_snr - msnr
                    # keep the best choice
                    temp_4_chan_index = np.max(np.where(temp_4_compare_snr == np.max(temp_4_compare_snr)))
                    temp_3_chan_index.append(str(temp_4_chan_index))
                    temp_3_compare_snr[l] = temp_4_compare_snr[temp_4_chan_index]
                # judge if there's improvement
                if np.max(temp_3_compare_snr) > np.max(snr_change):  # has improvement
                    # find index
                    delete_chan_index = np.max(np.where(temp_3_compare_snr == np.max(temp_3_compare_snr)))
                    add_chan_index = int(temp_3_chan_index[delete_chan_index])
                    # operate (refresh data)
                    del remain_chans[delete_chan_index]
                    remain_chans.append(chans[add_chan_index])
                    chan_index = add_chan_index
                    snr_change.append(temp_3_compare_snr[delete_chan_index])
                    # delete useless data & add new data
                    core_w = np.delete(core_w, delete_chan_index, axis=0)                   
                    temp_6_core_w = np.zeros((core_w.shape[0]+1, core_w.shape[1], core_w.shape[2]))
                    temp_6_core_w[:core_w.shape[0], :, :] = core_w
                    temp_6_core_w[core_w.shape[0], :, :] = w[:, add_chan_index, :]
                    core_w = copy.deepcopy(temp_6_core_w)
                    del temp_6_core_w               
                    core_data = np.delete(core_data, delete_chan_index, axis=0)              
                    temp_6_core_data = np.zeros((core_data.shape[0]+1, core_data.shape[1], core_data.shape[2]))
                    temp_6_core_data[:core_data.shape[0], :, :] = core_data
                    temp_6_core_data[core_data.shape[0], :, :] = signal_data[:, add_chan_index, :]
                    core_data = copy.deepcopy(temp_6_core_data)
                    del temp_6_core_data                
                    signal_data = np.delete(signal_data, add_chan_index, axis=1)
                    w = np.delete(w, add_chan_index, axis=1)
                    del chans[add_chan_index]
                    # release RAM
                    del temp_3_chans, temp_3_compare_snr, temp_3_chan_index
                    del temp_4_chans, temp_4_data, temp_4_w, temp_4_compare_snr, temp_4_chan_index
                    del temp_5_data, temp_5_w, temp_5_extract, mtemp_5_snr, temp_5_snr
                    # significant loop mark
                    print('Complete ' + str(j) + 'th loop')
                # no improvement
                else:
                    # release RAM
                    del temp_3_chans, temp_3_compare_snr, temp_3_chan_index
                    del temp_4_chans, temp_4_data, temp_4_w, temp_4_compare_snr, temp_4_chan_index
                    del temp_5_data, temp_5_w, temp_5_extract, mtemp_5_snr, temp_5_snr
                    # reset
                    print('Complete ' + str(j) + 'th loop')
                    print("Already best in " + str(j) + " channels' condition!")    
        j += 1    
    remain_chans = remain_chans[:len(remain_chans)-1]
    return remain_chans, snr_change

def stepwise_SRCA_fs(chans, msnr, w, w_target, signal_data, data_target, regression):
    '''
    Parameters:
        chans: list of channels; the list order corresponds to the data array's
        mfs: float; the mean of original signal's fisher score
        w: background part input data array (n_events, n_trials, n_chans, n_times)
        w_target: background part target data array (n_events, n_trials, n_times)
        signal_data: signal part input data array (n_events, n_trials, n_chans, n_times)
        data_target: signal part target data array (n_events, n_trials, n_times)
    Returns:
        model_chans: list of channels which should be used in MCEE
        para_change: list of SNR's alteration
    '''
    print('Running Stepwise SRCA...')
    start = time.clock()   
    j = 1   
    max_loop = len(chans)   
    remain_chans = []
    snr_change = []
    core_data = []
    core_w = []
    active = True
    n_trials = w.shape[1]
    n_times_w = w.shape[-1]
    n_times_s = signal_data.shape[-1]
    while active and len(chans) <= max_loop:
        compare_snr = np.zeros((len(chans)))
        mtemp_snr = np.zeros((len(chans)))
        for i in range(len(chans)):
            temp_extract = np.zeros_like(data_target)
            if j == 1:
                temp_w = w[:, :, i,:]
                temp_data = signal_data[:, :, i, :]
                for kk in range(2):
                    temp_extract[kk, :, :] = mlr(temp_w[kk, :, :], w_target[kk, :, :],
                    temp_data[kk, :, :], data_target[kk, :, :], regression)
            else:
                temp_w = np.zeros((j, 2, n_trials, n_times_w))
                temp_w[:-1, :, :, :] = core_w
                temp_w[-1, :, :, :] = w[:, :, i, :]                
                temp_data = np.zeros((j, 2, n_trials, n_times_s))
                temp_data[:-1, :, :, :] = core_data
                temp_data[-1, :, :, :] = signal_data[:, :, i, :]
                for kk in range(2):
                    temp_extract[kk, :, :] = mlr(temp_w[:, kk, :, :], w_target[kk, :, :],
                    temp_data[:, kk, :, :], data_target[kk, :, :], regression)
            mtemp_snr[i] = np.mean(fisher_score(temp_extract))
            compare_snr[i] = mtemp_snr[i] - msnr
        chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
        remain_chans.append(chans.pop(chan_index))
        snr_change.append(np.max(compare_snr))    
        if j == 1: 
            core_w = w[:, :, chan_index, :]
            core_data = signal_data[:, :, chan_index, :]
            signal_data = np.delete(signal_data, chan_index, axis=2)
            w = np.delete(w ,chan_index, axis=2)
            print('Complete ' + str(j) + 'th loop')         
        if j == 2:  
            temp_core_w = np.zeros((j, 2, n_trials, n_times_w))
            temp_core_w[0, :, :, :] = core_w
            temp_core_w[1, :, :, :] = w[:, :, chan_index, :]
            core_w = copy.deepcopy(temp_core_w)
            temp_core_data = np.zeros((j, 2, n_trials, n_times_s))
            temp_core_data[0, :, :, :] = core_data
            temp_core_data[1, :, :, :] = signal_data[:, :, chan_index, :]
            core_data = copy.deepcopy(temp_core_data)
            if snr_change[-1] < np.max(snr_change):
                print('Stepwise EE complete!')
                end = time.clock()
                print('Recursive running time: ' + str(end - start) + 's')
                active = False
            else:
                signal_data = np.delete(signal_data, chan_index, axis=2)
                w = np.delete(w, chan_index, axis=2)
                temp_1_chans = copy.deepcopy(remain_chans)
                temp_1_data = copy.deepcopy(core_data)
                temp_1_w = copy.deepcopy(core_w)
                del temp_1_chans[0]
                temp_1_data = np.delete(temp_1_data, 0, axis=0)
                temp_1_w = np.delete(temp_1_w, 0, axis=0)
                temp_2_compare_snr = np.zeros((signal_data.shape[2]))
                for k in range(signal_data.shape[2]):
                    temp_2_w = np.zeros((2, 2, n_trials, n_times_w))
                    temp_2_w[0, :, :, :] = temp_1_w
                    temp_2_w[1, :, :, :] = w[:, :, k, :]
                    temp_2_data = np.zeros((2, 2, n_trials, n_times_s))
                    temp_2_data[0, :, :, :] = temp_1_data
                    temp_2_data[1, :, :, :] = signal_data[:, :, k, :]
                    temp_2_extract = np.zeros_like(data_target)
                    for kk in range(2):
                        temp_2_extract[kk, :, :] = mlr(temp_2_w[:, kk, :, :],
                            w_target[kk, :, :], temp_2_data[:, kk, :, :],
                            data_target[kk, :, :], regression)
                    mtemp_2_snr = np.mean(fisher_score(temp_2_extract))
                    temp_2_compare_snr[k] = mtemp_2_snr - msnr
                temp_2_chan_index = np.max(np.where(temp_2_compare_snr == np.max(temp_2_compare_snr)))
                if temp_2_compare_snr[temp_2_chan_index] > snr_change[-1]:
                    chan_index = temp_2_chan_index
                    remain_chans.append(chans.pop(chan_index))
                    snr_change.append(temp_2_compare_snr[temp_2_chan_index])
                    core_w = np.delete(core_w, 0, axis=0)
                    temp_2_core_w = np.zeros((2, 2, n_trials, n_times_w))
                    temp_2_core_w[0, :, :, :] = core_w
                    temp_2_core_w[1, :, :, :] = w[:, :, chan_index, :]
                    core_w = copy.deepcopy(temp_2_core_w)
                    core_data = np.delete(core_data, 0, axis=0)
                    temp_2_core_data = np.zeros((2, 2, n_trials, n_times_s))
                    temp_2_core_data[0, :, :, :] = core_data
                    temp_2_core_data[1, :, :, :] = signal_data[:, :, chan_index, :]
                    core_data = copy.deepcopy(temp_2_core_data)
                    signal_data = np.delete(signal_data, chan_index, axis=2)
                    w = np.delete(w, chan_index, axis=2)
                    del remain_chans[0]
                    print('Complete ' + str(j) + 'th loop')               
                else:
                    print("Already best in 2 channels' contidion!")
        if j > 2:
            temp_core_w = np.zeros((j, 2, n_trials, n_times_w))
            temp_core_w[:-1, :, :, :] = core_w
            temp_core_w[-1, :, :, :] = w[:, :, chan_index, :]
            core_w = copy.deepcopy(temp_core_w)
            temp_core_data = np.zeros((j, 2, n_trials, n_times_s))
            temp_core_data[:-1, :, :, :] = core_data
            temp_core_data[-1, :, :, :] = signal_data[:, :, chan_index, :]
            core_data = copy.deepcopy(temp_core_data)
            if snr_change[-1] < np.max(snr_change):
                print('Stepwise EE complete!')
                end = time.clock()
                print('Recursive running time: ' + str(end - start) + 's')
                active = False
            else:
                signal_data = np.delete(signal_data, chan_index, axis=2)
                w = np.delete(w, chan_index, axis=2)
                temp_3_chans = copy.deepcopy(remain_chans)
                temp_3_compare_snr = np.zeros((len(temp_3_chans)-1))
                temp_3_chan_index = []
                for l in range(len(temp_3_chans)-1):
                    temp_4_chans = copy.deepcopy(remain_chans)
                    temp_4_data = copy.deepcopy(core_data)
                    temp_4_w = copy.deepcopy(core_w)
                    del temp_4_chans[l]
                    temp_4_data = np.delete(temp_4_data, l, axis=0)
                    temp_4_w = np.delete(temp_4_w, l, axis=0)
                    temp_4_compare_snr = np.zeros((signal_data.shape[2]))
                    for m in range(signal_data.shape[2]):
                        temp_5_w = np.zeros((j, 2, n_trials, n_times_w))
                        temp_5_w[:-1, :, :, :] = temp_4_w
                        temp_5_w[-1, :, :, :] = w[:, :, m, :]               
                        temp_5_data = np.zeros((j, 2, n_trials, n_times_s))
                        temp_5_data[:-1, :, :, :] = temp_4_data
                        temp_5_data[-1, :, :, :] = signal_data[:, :, m, :]
                        temp_5_extract = np.zeros_like(data_target)
                        for kk in range(2):
                            temp_5_extract[kk, :, :] = mlr(temp_5_w[:, kk, :, :],
                                w_target[kk, :, :], temp_5_data[:, kk, :, :],
                                data_target[kk, :, :], regression)
                        mtemp_5_snr = np.mean(fisher_score(temp_5_extract))
                        temp_4_compare_snr[m] = mtemp_5_snr - msnr
                    temp_4_chan_index = np.max(np.where(temp_4_compare_snr == np.max(temp_4_compare_snr)))
                    temp_3_chan_index.append(str(temp_4_chan_index))
                    temp_3_compare_snr[l] = temp_4_compare_snr[temp_4_chan_index]
                if np.max(temp_3_compare_snr) > np.max(snr_change):
                    delete_chan_index = np.max(np.where(temp_3_compare_snr == np.max(temp_3_compare_snr)))
                    add_chan_index = int(temp_3_chan_index[delete_chan_index])
                    del remain_chans[delete_chan_index]
                    remain_chans.append(chans[add_chan_index])
                    chan_index = add_chan_index
                    snr_change.append(temp_3_compare_snr[delete_chan_index])
                    core_w = np.delete(core_w, delete_chan_index, axis=0)                   
                    temp_6_core_w = np.zeros((core_w.shape[0]+1, 2, core_w.shape[2], core_w.shape[-1]))
                    temp_6_core_w[:-1, :, :, :] = core_w
                    temp_6_core_w[-1, :, :, :] = w[:, :, add_chan_index, :]
                    core_w = copy.deepcopy(temp_6_core_w)
                    core_data = np.delete(core_data, delete_chan_index, axis=0)              
                    temp_6_core_data = np.zeros((core_data.shape[0]+1, 2, core_data.shape[2], core_data.shape[-1]))
                    temp_6_core_data[:-1, :, :, :] = core_data
                    temp_6_core_data[-1, :, :, :] = signal_data[:, :, add_chan_index, :]
                    core_data = copy.deepcopy(temp_6_core_data)
                    signal_data = np.delete(signal_data, add_chan_index, axis=2)
                    w = np.delete(w, add_chan_index, axis=2)
                    del chans[add_chan_index]
                    print('Complete ' + str(j) + 'th loop')
                else:
                    print('Complete ' + str(j) + 'th loop')
                    print("Already best in " + str(j) + " channels' condition!")    
        j += 1    
    remain_chans = remain_chans[:len(remain_chans)-1]
    return remain_chans, snr_change

#%% Target identification: TRCA method
def fbtrca(tr_fb_data, te_fb_data):
    '''
    TRCA is the method that extracts task-related components efficiently 
        by maximizing the reproducibility during the task period
    Parameters:
        tr_fb_data: (n_events, n_bands, n_trials, n_chans, n_times) |
            training dataset (after filter bank) 
        te_fb_data: (n_events, n_bands, n_trials, n_chans, n_times) |
            test dataset (after filter bank)
    Returns:
        accuracy: int | the number of correct identifications
        
    '''
    # template data: (n_events, n_bands, n_chans, n_times)|basic element: (n_chans, n_times)
    template = np.mean(tr_fb_data, axis=2)  
    # basic parameters
    n_events = tr_fb_data.shape[0]
    n_bands = tr_fb_data.shape[1]
    n_chans = tr_fb_data.shape[3]
    n_times = tr_fb_data.shape[4]   
    # Matrix Q: inter-channel covariance
    q = np.zeros((n_events, n_bands, n_chans, n_chans))
    # all events(n), all bands(m)
    for x in range(n_events):  # x for events (n)
        for y in range(n_bands):  # y for bands (m)
            temp = np.zeros((n_chans, int(tr_fb_data.shape[2]*n_times)))
            for z in range(n_chans):  # z for channels
                # concatenated matrix of all trials in training dataset
                temp[z,:] = tr_fb_data[x,y,:,z,:].flatten()
            # compute matrix Q | (Toeplitz matrix): (n_chans, n_chans)
            # for each event & band, there should be a unique Q
            # so the total quantity of Q is n_bands*n_events (here is 30=x*y)
            q[x,y,:,:] = np.cov(temp)
            del temp, z
        del y
    del x
    # Matrix S: inter-channels' inter-trial covariance
    # all events(n), all bands(m), inter-channel(n_chans, n_chans)
    s = np.zeros((n_events, n_bands, n_chans, n_chans))
    for u in range(n_events):  # u for events
        for v in range(n_bands):  # v for bands
            # at the inter-channels' level, obviouly the square will also be a Toeplitz matrix
            # i.e. (n_chans, n_chans), here the shape of each matrix should be (9,9)
            for w in range(n_chans):  # w for channels (j1)
                for x in range(n_chans):  # x for channels (j2)
                    cov = []
                    # for each event & band & channel, there should be (trials^2-trials) values
                    # here trials = 10, so there should be 90 values in each loop
                    for y in range(tr_fb_data.shape[2]):  # y for trials (h1)
                        temp = np.zeros((2, n_times))
                        temp[0,:] = tr_fb_data[u,v,y,w,:]
                        for z in range(tr_fb_data.shape[2]):  # z for trials (h2)
                            if z != y:  # h1 != h2, INTER-trial covariance
                                temp[1,:] = tr_fb_data[u,v,z,x,:]
                                cov.append(np.sum(np.tril(np.cov(temp), -1)))
                            else:
                                continue
                        del z, temp
                    del y
                    # the basic element S(j1j2) of Matrix S
                    # is the sum of inter-trial covariance (h1&h2) of 1 events & 1 band in 1 channel
                    # then form a square (n_chans,n_chans) to describe inter-channels' information
                    # then form a data cube containing various bands and events' information      
                    # of course the 1st value should be the larger one (computed in 1 channel)
                    # according to the spatial location of different channels
                    # there should also be size differences
                    # (e.g. PZ & POZ's values are significantly larger)
                    s[u,v,w,x] = np.sum(cov)
                    del cov
                del x
            del w
        del v
    del u   
    # Spatial filter W
    # all events(n), all bands(m)
    w = np.zeros((n_events, n_bands, n_chans))
    for y in range(n_events):
        for z in range(n_bands):
            # Square Q^-1 * S
            qs = np.mat(q[y,z,:,:]).I * np.mat(s[y,z,:,:])
            # Eigenvalues & eigenvectors
            e_value, e_vector = np.linalg.eig(qs)
            # choose the eigenvector which refers to the largest eigenvalue
            w_index = np.max(np.where(e_value == np.max(e_value)))
            # w will maximum the task-related componont from multi-channel's data
            w[y,z,:] = e_vector[:,w_index].T
            del w_index
        del z
    del y
    # from now on, never use w as loop mark because we have variable named w
    # Test dataset operating
    # basic element of r is (n_bands, n_events)
    r = np.zeros((n_events, te_fb_data.shape[2], n_bands, n_events))
    for v in range(n_events): # events in test dataset
        for x in range(te_fb_data.shape[2]):  # trials in test dataset (of one event)
            for y in range(n_bands):  # bands are locked
                # (vth event, zth band, xth trial) test data to (all events(n), zth band(m)) training data
                for z in range(n_events):
                    temp_test = np.mat(te_fb_data[v,y,x,:,:]).T * np.mat(w[z,y,:]).T
                    temp_template = np.mat(template[z,y,:,:]).T * np.mat(w[z,y,:]).T
                    r[v,x,y,z] = np.sum(np.tril(np.corrcoef(temp_test.T, temp_template.T),-1))
                del z, temp_test, temp_template
            del y
        del x
    del v
    # Feature for target identification
    r = r**2
    # identification function a(m)
    a = np.matrix([(m+1)**-1.25+0.25 for m in range(n_bands)])
    rou = np.zeros((n_events, te_fb_data.shape[2], n_events))
    for y in range(n_events):
        for z in range(te_fb_data.shape[2]):  # trials in test dataset (of one event)
            # (yth event, zth trial) test data | will have n_events' value, here is 3
            # the location of the largest value refers to the class of this trial
            rou[y,z,:] = a * np.mat(r[y,z,:,:])    
    acc = []
    # compute accuracy
    for x in range(rou.shape[0]):  # ideal classification
        for y in range(rou.shape[1]):
            if np.max(np.where(rou[x,y,:] == np.max(rou[x,y,:]))) == x:  # correct
                acc.append(1)   
    return acc

def pure_trca(train_data, test_data):
    '''
    TRCA without filter banks
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
    Returns:
        accuracy: int | the number of correct identifications
    '''
    # template data: (n_events, n_chans, n_times) | basic element: (n_chans, n_times)
    template = np.mean(train_data, axis=1)    
    # basic parameters
    n_events = train_data.shape[0]
    n_chans = train_data.shape[2]
    n_times = train_data.shape[3]   
    # Matrix Q: inter-channel covariance
    q = np.zeros((n_events, n_chans, n_chans))
    for x in range(n_events):
        temp = np.zeros((n_chans, int(train_data.shape[1]*n_times)))
        for y in range(n_chans):
            # concatenated matrix of all trials in training dataset
            temp[y,:] = train_data[x,:,y,:].flatten()
        del y
        q[x,:,:] = np.cov(temp)
        del temp
    del x
    print('Matrix Q complete!')
    # Matrix S: inter-channels' inter-trial covariance
    s = np.zeros((n_events, n_chans, n_chans))
    for v in range(n_events):  # v for events
        for w in range(n_chans):  # w for channels (j1)
            for x in range(n_chans):  # x for channels (j2)
                cov = []
                for y in range(train_data.shape[1]):  # y for trials (h1)
                    temp = np.zeros((2, n_times))
                    temp[0,:] = train_data[v,y,w,:]
                    for z in range(train_data.shape[1]):  # z for trials (h2)
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
    print('Matrix S complete!')  
    # Spatial filter W
    w = np.zeros((n_events, n_chans))
    for z in range(n_events):
        # Square Q^-1 * S
        qs = np.mat(q[z,:,:]).I * np.mat(s[z,:,:])
        # Eigenvalues & eigenvectors
        e_value, e_vector = np.linalg.eig(qs)
        # choose the eigenvector refering to the largest eigenvalue
        w_index = np.max(np.where(e_value == np.max(e_value)))
        w[z,:] = e_vector[:,w_index].T
        del w_index
    del z
    print('Spatial filter complete!')   
    # Test dataset operating
    r = np.zeros((n_events, test_data.shape[1], n_events))
    for x in range(n_events):  # n_events in test dataset
        for y in range(test_data.shape[1]):
            for z in range(n_events):
                temp_test = np.mat(test_data[x,y,:,:]).T * np.mat(w[z,:]).T
                temp_template = np.mat(template[z,:,:]).T * np.mat(w[z,:]).T
                r[x,y,z] = np.sum(np.tril(np.corrcoef(temp_test.T, temp_template.T),-1))
            del z, temp_test, temp_template
        del y
    del x
    print('Target identification complete!')
    # Compute accuracy
    acc = []
    for x in range(r.shape[0]):  # ideal classification
        for y in range(r.shape[1]):
            if np.max(np.where(r[x,y,:] == np.max(r[x,y,:]))) == x:  # correct
                acc.append(1)               
    return acc
    
def e_srca_trca(train_data, test_data, tar_chans, model_chans, chans,
                regression, sp):
    '''
    ensemble-TRCA without filter banks
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
    Returns:
        accuracy: int | the number of correct identifications
    '''  
    # config correct srca process on training dataset
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    for ntc in range(len(tar_chans)):
        target_channel = tar_chans[ntc]
        for ne in range(n_events):
            model_chan = model_chans[ntc*2+ne]
            w_i = np.zeros((n_trains, len(model_chan), 1000))
            sig_i = np.zeros((n_trains, len(model_chan), n_times))
            for nc in range(len(model_chan)):
                w_i[:, nc, :] = train_data[ne, :, chans.index(model_chan[nc]), :1000]
                sig_i[:, nc, :] = train_data[ne, :, chans.index(model_chan[nc]), sp:]
            del nc
            w_o = train_data[ne, :, chans.index(target_channel), :1000]
            sig_o = train_data[ne, :, chans.index(target_channel), sp:]            
            w_i = np.swapaxes(w_i, 0, 1)
            sig_i = np.swapaxes(sig_i, 0, 1)
            w_ex_s = mlr(model_input=w_i, model_target=w_o, data_input=sig_i,
                         data_target=sig_o, regression=regression)
            model_sig[ne,:,ntc,:] = w_ex_s
        del ne
    del ntc
    del n_trains
    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for ntc in range(len(tar_chans)):
        target_channel = tar_chans[ntc]
        for ne_tr in range(n_events):
            model_chan = model_chans[ntc*2+ne_tr]
            for ne_re in range(n_events):
                w_i = np.zeros((n_tests, len(model_chan), 1000))
                sig_i = np.zeros((n_tests, len(model_chan), n_times))
                for nc in range(len(model_chan)):
                    w_i[:, nc, :] = test_data[ne_re, :, chans.index(model_chan[nc]), :1000]
                    sig_i[:, nc, :] = test_data[ne_re, :, chans.index(model_chan[nc]), sp:]
                del nc
                w_o = test_data[ne_re, :, chans.index(target_channel), :1000]
                sig_o = test_data[ne_re, :, chans.index(target_channel), sp:]
                w_i = np.swapaxes(w_i, 0, 1)
                sig_i = np.swapaxes(sig_i, 0, 1)
                w_ex_s = mlr(model_input=w_i, model_target=w_o, data_input=sig_i,
                         data_target=sig_o, regression=regression)
                target_sig[ne_tr, ne_re, :, ntc, :] = w_ex_s
            del ne_re
        del ne_tr
    del ntc
    del n_tests, n_events, n_chans, n_times
    # template data: (n_events, n_chans, n_times)
    template = np.mean(model_sig, axis=1)
    # basic parameters
    n_events = model_sig.shape[0]
    n_chans = model_sig.shape[2]
    n_times = model_sig.shape[-1]
    n_trials = model_sig.shape[1]
    # Matrix Q: inter-channel covariance
    q = np.zeros((n_events, n_chans, n_chans))
    for x in range(n_events):
        temp = np.zeros((n_chans, int(n_trials*n_times)))
        for y in range(n_chans):
            temp[y,:] = model_sig[x,:,y,:].flatten()
        del y
        q[x,:,:] = np.cov(temp)
        del temp
    del x
    # Matrix S: inter-channels' inter-trial covariance
    s = np.zeros((n_events, n_chans, n_chans))
    for v in range(n_events):  # v for events
        for w in range(n_chans):  # w for channels (j1)
            for x in range(n_chans):  # x for channels (j2)
                cov = []
                for y in range(model_sig.shape[1]):  # y for trials (h1)
                    temp = np.zeros((2, n_times))
                    temp[0,:] = model_sig[v,y,w,:]
                    for z in range(model_sig.shape[1]):  # z for trials (h2)
                        if z != y:  # h1 != h2
                            temp[1,:] = model_sig[v,z,x,:]
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
    # Spatial filter W
    w = np.zeros((n_events, n_chans))
    for z in range(n_events):
        # Square Q^-1 * S
        qs = np.mat(q[z,:,:]).I * np.mat(s[z,:,:])
        # Eigenvalues & eigenvectors
        e_value, e_vector = np.linalg.eig(qs)
        # choose the eigenvector refering to the largest eigenvalue
        w_index = np.max(np.where(e_value == np.max(e_value)))
        w[z,:] = e_vector[:,w_index].T
        del w_index
    del z
    del n_events, n_chans, n_times, n_trials
    # Ensemble target identification
    n_ev_srca = target_sig.shape[0]
    n_ev_real = target_sig.shape[1]
    n_trials = target_sig.shape[2]
    r = np.zeros((n_ev_srca, n_ev_real, n_trials))
    for nes in range(n_ev_srca):  # n_events in srca model
        for ner in range(n_ev_real):  # n_events in test dataset
            for nt in range(n_trials):  # n_trials
                temp_test = (np.mat(w) * np.mat(target_sig[nes, ner, nt, :, :])).flatten()
                temp_template = (np.mat(w) * np.mat(template[nes, :, :])).flatten()
                r[nes, ner, nt] = np.sum(np.tril(np.corrcoef(temp_test, temp_template), -1))
            del nt
        del ner
    del nes
    del n_ev_srca, n_ev_real
    # Compute accuracy
    acc = []
    for z in range(n_trials):
        if r[0,0,z] > r[1,0,z]:
            acc.append(1)
    for z in range(n_trials):
        if r[1,1,z] > r[0,1,z]:
            acc.append(1)
    del n_trials
    return acc

def e_trca(train_data, test_data):
    '''
    ensemble-TRCA without filter banks for original signal
    Parameters:
        train_data: (n_events, n_trials, n_chans, n_times) | training dataset
        test_data: (n_events, n_trials, n_chans, n_times) | test dataset
    Returns:
        accuracy: int | the number of correctt identification
    '''
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_chans = train_data.shape[2]
    n_times = train_data.shape[-1]
    
    template = np.mean(train_data, axis=1)
    
    q = np.zeros((n_events, n_chans, n_chans))
    for x in range(n_events):
        temp = np.zeros((n_chans, int(n_trains*n_times)))
        for y in range(n_chans):
            temp[y,:] = train_data[x,:,y,:].flatten()
        del y
        q[x,:,:] = np.cov(temp)
        del temp
    del x
    
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
    
    w = np.zeros((n_events, n_chans))
    for z in range(n_events):
        qs = np.mat(q[z,:,:]).I * np.mat(s[z,:,:])
        e_value, e_vector = np.linalg.eig(qs)
        w_index = np.max(np.where(e_value == np.max(e_value)))
        w[z,:] = e_vector[:,w_index].T
        del w_index
    del z
    
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
    
    acc = []
    for x in range(r.shape[0]):
        for y in range(r.shape[1]):
            if np.max(np.where(r[x,y,:] == np.max(r[x,y,:]))) == x:
                acc.append(1)
    return acc

def srca_trca(train_data, test_data, tar_chans, model_chans, chans,
                regression, sp):
    '''
    '''
    # config correct srca process on training dataset
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_chans = len(tar_chans)
    n_times = train_data.shape[-1] - sp
    model_sig = np.zeros((n_events, n_trains, n_chans, n_times))
    for ntc in range(len(tar_chans)):
        target_channel = tar_chans[ntc]
        for ne in range(n_events):
            model_chan = model_chans[ntc*2+ne]
            w_i = np.zeros((n_trains, len(model_chan), 1000))
            sig_i = np.zeros((n_trains, len(model_chan), n_times))
            for nc in range(len(model_chan)):
                w_i[:, nc, :] = train_data[ne, :, chans.index(model_chan[nc]), :1000]
                sig_i[:, nc, :] = train_data[ne, :, chans.index(model_chan[nc]), sp:]
            del nc
            w_o = train_data[ne, :, chans.index(target_channel), :1000]
            sig_o = train_data[ne, :, chans.index(target_channel), sp:]            
            w_i = np.swapaxes(w_i, 0, 1)
            sig_i = np.swapaxes(sig_i, 0, 1)
            w_ex_s = mlr(model_input=w_i, model_target=w_o, data_input=sig_i,
                         data_target=sig_o, regression=regression)
            model_sig[ne,:,ntc,:] = w_ex_s
        del ne
    del ntc
    del n_trains
    # apply different srca models on one trial's data
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for ntc in range(len(tar_chans)):
        target_channel = tar_chans[ntc]
        for ne_tr in range(n_events):
            model_chan = model_chans[ntc*2+ne_tr]
            for ne_re in range(n_events):
                w_i = np.zeros((n_tests, len(model_chan), 1000))
                sig_i = np.zeros((n_tests, len(model_chan), n_times))
                for nc in range(len(model_chan)):
                    w_i[:, nc, :] = test_data[ne_re, :, chans.index(model_chan[nc]), :1000]
                    sig_i[:, nc, :] = test_data[ne_re, :, chans.index(model_chan[nc]), sp:]
                del nc
                w_o = test_data[ne_re, :, chans.index(target_channel), :1000]
                sig_o = test_data[ne_re, :, chans.index(target_channel), sp:]
                w_i = np.swapaxes(w_i, 0, 1)
                sig_i = np.swapaxes(sig_i, 0, 1)
                w_ex_s = mlr(model_input=w_i, model_target=w_o, data_input=sig_i,
                         data_target=sig_o, regression=regression)
                target_sig[ne_tr, ne_re, :, ntc, :] = w_ex_s
            del ne_re
        del ne_tr
    del ntc
    del n_tests, n_events, n_chans, n_times
    # template data: (n_events, n_chans, n_times)
    template = np.mean(model_sig, axis=1)
    # basic parameters
    n_events = model_sig.shape[0]
    n_chans = model_sig.shape[2]
    n_times = model_sig.shape[-1]
    n_trials = model_sig.shape[1]
    # Matrix Q: inter-channel covariance
    q = np.zeros((n_events, n_chans, n_chans))
    for x in range(n_events):
        temp = np.zeros((n_chans, int(n_trials*n_times)))
        for y in range(n_chans):
            temp[y,:] = model_sig[x,:,y,:].flatten()
        del y
        q[x,:,:] = np.cov(temp)
        del temp
    del x
    # Matrix S: inter-channels' inter-trial covariance
    s = np.zeros((n_events, n_chans, n_chans))
    for v in range(n_events):  # v for events
        for w in range(n_chans):  # w for channels (j1)
            for x in range(n_chans):  # x for channels (j2)
                cov = []
                for y in range(model_sig.shape[1]):  # y for trials (h1)
                    temp = np.zeros((2, n_times))
                    temp[0,:] = model_sig[v,y,w,:]
                    for z in range(model_sig.shape[1]):  # z for trials (h2)
                        if z != y:  # h1 != h2
                            temp[1,:] = model_sig[v,z,x,:]
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
    # Spatial filter W
    w = np.zeros((n_events, n_chans))
    for z in range(n_events):
        # Square Q^-1 * S
        qs = np.mat(q[z,:,:]).I * np.mat(s[z,:,:])
        # Eigenvalues & eigenvectors
        e_value, e_vector = np.linalg.eig(qs)
        # choose the eigenvector refering to the largest eigenvalue
        w_index = np.max(np.where(e_value == np.max(e_value)))
        w[z,:] = e_vector[:,w_index].T
        del w_index
    del z
    del n_events, n_chans, n_times, n_trials
    # target identification
    n_ev_srca = target_sig.shape[0]
    n_ev_real = target_sig.shape[1]
    n_trials = target_sig.shape[2]
    r = np.zeros((n_ev_srca, n_ev_real, n_trials))
    for nes in range(n_ev_srca):  # n_events in srca model
        for ner in range(n_ev_real):  # n_events in test dataset
            for nt in range(n_trials):  # n_trials
                temp_test = np.mat(w[nes, :]) * np.mat(target_sig[nes, ner, nt, :, :])
                temp_template = np.mat(w[nes, :]) * np.mat(template[nes, :, :])
                r[nes, ner, nt] = np.sum(np.tril(np.corrcoef(temp_test, temp_template),-1))
            del nt
        del ner
    del nes, n_ev_srca, n_ev_real
    # compute accuracy
    acc = []
    for z in range(n_trials):
        if r[0,0,z] > r[1,0,z]:
            acc.append(1)
    for z in range(n_trials):
        if r[1,1,z] > r[0,1,z]:
            acc.append(1)
    del n_trials
    return acc

#%% Correlation detect for single-channel data
def corr_detect(test_data, template):
    '''
    Offline Target identification for single-channel data
        (using Pearson correlation coefficients)
    Parameters:
        test_data: array | (n_events, n_trials, n_times)
        template: array | (n_events, n_times)
    Returns:
        acc: int | the total number of correct identifications
        mr: square | (n_events (test dataset), n_events (template)),
            the mean of Pearson correlation coefficients between N th events'
            test data and M th events' template
        rou: array | (n_events (test dataset), n_trials, n_events (template))
            details of mr
    '''
    # initialization
    n_events = template.shape[0]
    n_trials = test_data.shape[1]
    acc = []
    mr = np.zeros((n_events, n_events))
    rou = np.zeros((n_events, n_trials, n_events))   
    # compute Pearson correlation coefficients & target identification
    for nete in range(n_events):  # n_events' loop in test dataset
        for ntte in range(n_trials):  # n_trials' loop in test dataset
            for netr in range(n_events):  # n_events' loop in template (training dataset)
                rou[nete,ntte,netr] = np.sum(np.tril(np.corrcoef(test_data[nete,ntte,:],template[netr,:]),-1))
            del netr
            if np.max(np.where([rou[nete,ntte,:] == np.max(rou[nete,ntte,:])])) == nete:  # correct
                acc.append(1)
        del ntte
    del nete
    acc = np.sum(acc) 
    for j in range(n_events):
        for k in range(n_events):
            mr[j,k] = np.mean(rou[j,:,k])
            
    return acc, mr, rou