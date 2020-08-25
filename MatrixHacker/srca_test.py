# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:57:54 2020

@author: yiwei
"""

import numpy as np
from matrixhacker.datasets.weibo2014 import Weibo2014
from matrixhacker.algorithms.utils.filtering import OnlineBlockFilter
import matrixhacker.algorithms.manifold.riemann as RGC
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_mi_mat_data(subject,limb_channels):
    all_channels=Weibo2014()._CHANNELS
    sub_raw_dict = Weibo2014()._get_single_subject_data(subject)
    sub_raw_array = sub_raw_dict['session_0']['run_0']
    sub_choice_array=np.concatenate((np.array(sub_raw_array[0:57][0]),
                                     np.array(sub_raw_array[58:61][0])))
    sub_label=np.squeeze(sub_raw_array[-1][0])
    sub_time=np.squeeze(sub_raw_array[-1][1])
    n_trial=0
    for i in range(len(sub_label)):
        if sub_label[i]==0:
            sub_label[i]=sub_label[i-1]
        else:
            n_trial=n_trial+1
    feet_time_i = np.array(np.where(sub_label==4)[0])
    rest_time_i = np.array(np.where(sub_label==7)[0])
    trial_len = int(len(sub_time)/n_trial)
    raw_trial = np.array(np.split(sub_choice_array,n_trial,axis=1))
    feet_trial=raw_trial[(feet_time_i[np.squeeze((np.array(np.where(feet_time_i%1600==0))))]/1600).astype(int)]
    rest_trial=raw_trial[(rest_time_i[np.squeeze((np.array(np.where(rest_time_i%1600==0))))]/1600).astype(int)]
    limb_index=np.zeros(len(limb_channels))
    for i in range(len(limb_channels)):
        limb_index[i]=all_channels.index(limb_channels[i])
    limb_index=limb_index.astype(int)
    feet_data_x=feet_trial[0:len(feet_trial),limb_index,int(trial_len*3/8):int(trial_len*7/8)]
    rest_data_x=rest_trial[0:len(rest_trial),limb_index,int(trial_len*3/8):int(trial_len*7/8)]
    feet_data_y=(np.zeros(len(feet_data_x))).astype(int)
    rest_data_y=(np.ones(len(rest_data_x))).astype(int)
    return feet_data_x,feet_data_y,rest_data_x,rest_data_y

def cov_matrix(data):
    data=data.copy()
    if data.ndim==2:
        for i in range(len(data)):
            data_i_mean=np.mean(data[i])
            data[i]=data[i]-data_i_mean
        cov=(1/(data.shape[1]-1))*np.dot(data,data.T)
    if data.ndim==3:
        cov=np.zeros((data.shape[0],data.shape[1],data.shape[1]))
        for i in range(len(data)):
            for j in range(len(data[0])):
                data_ij_mean=np.mean(data[i][j])
                data[i][j]=data[i][j]-data_ij_mean
            cov[i]=(1/(data.shape[2]-1))*np.dot(data[i],data[i].T)
    return cov

def fisher_score(data1,data2):
    '''
    data1: (n_trails, n_times)
    data2: (n_trails, n_times)
    '''
    miu1 = np.mean(data1,axis=0)
    miu2 = np.mean(data2,axis=0) 
    itr_d=np.dot((miu1-miu2),(miu1-miu2).T)
    ite_d1=np.sum(np.diag(np.dot((data1-miu1),(data1-miu1).T)))
    ite_d2=np.sum(np.diag(np.dot((data2-miu2),(data2-miu2).T)))
    fs = (ite_d1+ite_d2) /itr_d
    return fs

limb_channels=['F3','F1','FZ','F2','F4','FC3','FC1','FCZ','FC2','FC4','C3','C1','C2','C4','CP3','CP1','CPZ','CP2','CP4','P3','P1','PZ','P2','P4','CZ']
target_channel='CZ'
tar_index=limb_channels.index(target_channel)
srate=200
Online_filter=OnlineBlockFilter(srate=srate,filters=[[4,30]])
Online_filter.fit()
k=1
feet_data_x,feet_data_y,rest_data_x,rest_data_y=get_mi_mat_data(k,limb_channels)
y=np.concatenate((feet_data_y,rest_data_y))
raw_X=np.concatenate((feet_data_x,rest_data_x))
filter_X=np.squeeze(Online_filter.transform(raw_X))
index=np.arange(len(y))

kf=KFold(n_splits=2, shuffle=True, random_state=None)

for train_index,test_index in kf.split(filter_X):
    xtrain=filter_X[train_index]
    ytrain=y[train_index]
    rest_index=np.squeeze(np.array(np.where(ytrain==0)))
    foot_index=np.squeeze(np.array(np.where(ytrain==1)))
    restall_xtrain=xtrain[rest_index]
    footall_xtrain=xtrain[foot_index]
    resttar_xtrain=restall_xtrain[:,tar_index,:]
    foottar_xtrain=footall_xtrain[:,tar_index,:]
    


''' 
rest_data=restall_xtrain
foot_data=footall_xtrain
channels=deepcopy(limb_channels)

init_w=np.array([])
init_fun=float('inf')
temp_ch_i=0


while channels[temp_ch_i]!=channels[-1]:
    rest_tar=rest_data[:,-1,:]
    foot_tar=foot_data[:,-1,:]
    rest_oth=np.delete(rest_data,-1,axis=1)
    foot_oth=np.delete(foot_data,-1,axis=1)
    def func(w, sign=1.0):
        rest_com=np.zeros((rest_oth.shape[0],rest_oth.shape[-1]))
        foot_com=np.zeros((foot_oth.shape[0],foot_oth.shape[-1]))
        for i in range(len(rest_oth)):
            rest_com[i]=np.dot(w,rest_oth[i])
        for i in range(len(foot_oth)):
            foot_com[i]=np.dot(w,foot_oth[i])
        rest_exc=rest_tar-rest_com
        foot_exc=foot_tar-foot_com
    
        return sign*(fisher_score(rest_exc, foot_exc))
    
    cons = ({'type': 'eq','fun': lambda w: (np.sum(w)-1)},
            {'type': 'ineq','fun': lambda w: np.min(w)})
    
    opt_result = minimize(func, np.ones(len(channels)-1)/(len(channels)-1), 
                   constraints=cons, method='SLSQP', options={'disp': True})
    
    temp_w=opt_result.x
    temp_fun=opt_result.fun
    print(temp_fun)
    
    if temp_fun<init_fun:
        init_fun=temp_fun
        init_w=temp_w
        temp_ch_i=temp_ch_i+1

    else:
        rest_data=np.delete(rest_data,temp_ch_i,axis=1)
        foot_data=np.delete(foot_data,temp_ch_i,axis=1)
        del channels[temp_ch_i]

srca_chan_index=[]
for chan in channels:
    srca_chan_index.append(limb_channels.index(chan)) 

    print(temp_ch_i,rest_data.shape[1],rest_oth.shape[1],len(channels))
    print(init_fun,temp_fun)


print(temp_ch_i,rest_data.shape[1],rest_oth.shape[1],len(channels))


'''