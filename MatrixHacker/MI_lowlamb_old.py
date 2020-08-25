# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:39:34 2020

@author: yiwei
"""

import numpy as np
from matrixhacker.datasets.weibo2014 import Weibo2014
from matrixhacker.algorithms.utils.filtering import OnlineBlockFilter
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import matrixhacker.algorithms.manifold.riemann as RGC
from matrixhacker.algorithms.utils.base import (sqrtm, invsqrtm, powm)
import time
import joblib
import matplotlib.pyplot as plt

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
            cov[i]=(1/(data.shape[2]-1))*np.dot(data[j],data[j].T)
    return cov

class OnlineMDM(BaseEstimator, TransformerMixin, ClassifierMixin):
    
    def __init__(self,n_train,n_channel):
        """ Init."""
        self._mdm=RGC.MDM()
        self.n_channel=n_channel
        self.trial_num=0
        self.class_num=dict()
        self.n_train=n_train
        self.fit_X=np.zeros((self.n_train,self.n_channel,self.n_channel))
        self.fit_y=np.zeros(self.n_train).astype(int)
    
    def fit(self,new_X,new_y):
        if new_y in self.class_num:
            self.class_num[new_y]=self.class_num[new_y]+1
        else:
            self.class_num[new_y]=1
        if self.trial_num<self.n_train:
            self.fit_X[self.trial_num]=new_X
            self.fit_y[self.trial_num]=new_y
            if self.trial_num==self.n_train-1:
                self._mdm.fit(self.fit_X,self.fit_y)
            self.trial_num=self.trial_num+1
        return -1
    
    def static_strategy(self,new_X,new_y):
        self.trial_num=self.trial_num+1
        return self._mdm.predict(new_X)[0]
    
    def retrained_strategy(self,new_X,new_y):
        self.trial_num=self.trial_num+1
        predict_new_X=self._mdm.predict(new_X)
        new_X=new_X[np.newaxis,:]
        self.fit_X=np.concatenate((self.fit_X,new_X))
        self.fit_y=np.append(self.fit_y,new_y)
        self._mdm.fit(self.fit_X,self.fit_y)
        return predict_new_X[0]
    
    def incremental_strategy(self,new_X,new_y):
        self.trial_num=self.trial_num+1
        predict_new_X=self._mdm.predict(new_X)
        self.class_num[new_y]=self.class_num[new_y]+1
        if self._mdm.use_trace: new_X /= np.trace(new_X)
        temp_matrix=powm(np.dot(np.dot(invsqrtm(self._mdm.covmeans_[new_y]),new_X),invsqrtm(self._mdm.covmeans_[new_y])),
                         1/self.class_num[new_y])
        self._mdm.covmeans_[new_y]=np.dot(np.dot(sqrtm(self._mdm.covmeans_[new_y]),temp_matrix),
                                          sqrtm(self._mdm.covmeans_[new_y]))
        return predict_new_X[0]
    
    def running(self,new_X,new_y,strategy):
        new_X=new_X.copy()
        new_y=new_y.copy()
        if self.trial_num<self.n_train:
            return self.fit(new_X,new_y)
        else:
            if strategy=='static':
                return self.static_strategy(new_X,new_y)
            if strategy=='retrained':
                return self.retrained_strategy(new_X,new_y)
            if strategy=='incremental':
                return self.incremental_strategy(new_X,new_y)
                                
subject=1
limb_channels=['F3','FZ','F4','C3','CZ','C4','P3','PZ','P4']
#limb_channels=['F3','F1','FZ','F2','F4','FC3','FC1','FCZ','FC2','FC4','C3','C1','CZ','C2','C4','CP3','CP1','CPZ','CP2','CP4','P3','P1','PZ','P2','P4']
n_train=8
feet_data_x,feet_data_y,rest_data_x,rest_data_y=get_mi_mat_data(subject,limb_channels)
X=np.concatenate((feet_data_x,rest_data_x))
y=np.concatenate((feet_data_y,rest_data_y))
index=np.arange(len(y))
np.random.shuffle(index)
Online_filter=OnlineBlockFilter(srate=200,filters=[[4,13]])
Online_filter.fit()

ONC=OnlineMDM(n_train,n_channel=len(limb_channels))
static_result_data=[]
static_time_data=[]
for i in range(len(X)):
    time_start=time.perf_counter()
    new_X=cov_matrix(np.squeeze(Online_filter.transform(X[index[i]])))
    new_y=y[index[i]]
    result_i=ONC.running(new_X,new_y,'static')
    time_end=time.perf_counter()
    static_time_data.append(time_end-time_start)
    static_result_data.append(result_i)
    
ONC=OnlineMDM(n_train,n_channel=len(limb_channels))
retrained_result_data=[]
retrained_time_data=[]
for i in range(len(X)):
    time_start=time.perf_counter()
    new_X=cov_matrix(np.squeeze(Online_filter.transform(X[index[i]])))
    new_y=y[index[i]]
    result_i=ONC.running(new_X,new_y,'retrained')
    time_end=time.perf_counter()
    retrained_time_data.append(time_end-time_start)
    retrained_result_data.append(result_i)

ONC=OnlineMDM(n_train,n_channel=len(limb_channels))
incremental_result_data=[]
incremental_time_data=[]
for i in range(len(X)):
    time_start=time.perf_counter()
    new_X=cov_matrix(np.squeeze(Online_filter.transform(X[index[i]])))
    new_y=y[index[i]]
    result_i=ONC.running(new_X,new_y,'incremental')
    time_end=time.perf_counter()
    incremental_time_data.append(time_end-time_start)
    incremental_result_data.append(result_i)
