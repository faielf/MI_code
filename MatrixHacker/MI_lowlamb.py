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
from sklearn.metrics import accuracy_score
import time
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

all_sub_y=[]
all_sub_static_time=[]
all_sub_static_result=[]
all_sub_static_accuracy=[]
all_sub_retrained_time=[]
all_sub_retrained_result=[]
all_sub_retrained_accuracy=[]
all_sub_incremental_time=[]
all_sub_incremental_result=[]
all_sub_incremental_accuracy=[]
all_sub_rest_x=[]
all_sub_feet_x=[]
limb_channels=['F3','FZ','F4','C3','CZ','C4','P3','PZ','P4']
srate=200
n_train=8


for k in range(1,10):
    
    feet_data_x,feet_data_y,rest_data_x,rest_data_y=get_mi_mat_data(k,limb_channels)
    all_sub_feet_x.append(feet_data_x[:,4,:])
    all_sub_rest_x.append(rest_data_x[:,4,:])
    X=np.concatenate((feet_data_x,rest_data_x))
    y=np.concatenate((feet_data_y,rest_data_y))
    index=np.arange(len(y))
    np.random.shuffle(index)
    Online_filter=OnlineBlockFilter(srate=200,filters=[[4,30]])
    Online_filter.fit()
    all_sub_y.append(y[index[n_train:len(index)]])

    ONC=OnlineMDM(n_train,n_channel=len(limb_channels))    
    static_result_data=[]
    static_time_data=[]
    static_accuracy_data=[]
    for i in range(len(X)):
        time_start=time.perf_counter()
        new_X=cov_matrix(np.squeeze(Online_filter.transform(X[index[i]])))
        new_y=y[index[i]]
        result_i=ONC.running(new_X,new_y,'static')
        time_end=time.perf_counter()
        if i>=n_train:
            static_time_data.append((time_end-time_start)*1000)
            static_result_data.append(result_i)
            static_accuracy_data.append(accuracy_score(y[index[n_train:i+1]],static_result_data)*100)
    all_sub_static_time.append(static_time_data)
    all_sub_static_result.append(static_result_data)
    all_sub_static_accuracy.append(static_accuracy_data)
    
    ONC=OnlineMDM(n_train,n_channel=len(limb_channels))
    retrained_result_data=[]
    retrained_time_data=[]
    retrained_accuracy_data=[]
    
    for i in range(len(X)):
        time_start=time.perf_counter()
        new_X=cov_matrix(np.squeeze(Online_filter.transform(X[index[i]])))
        new_y=y[index[i]]
        result_i=ONC.running(new_X,new_y,'retrained')
        time_end=time.perf_counter()
        if i>=n_train:
            retrained_time_data.append((time_end-time_start)*1000)
            retrained_result_data.append(result_i)
            retrained_accuracy_data.append(accuracy_score(y[index[n_train:i+1]],retrained_result_data)*100)
    all_sub_retrained_time.append(retrained_time_data)
    all_sub_retrained_result.append(retrained_result_data)
    all_sub_retrained_accuracy.append(retrained_accuracy_data)
    
    ONC=OnlineMDM(n_train,n_channel=len(limb_channels))
    incremental_result_data=[]
    incremental_time_data=[]
    incremental_accuracy_data=[]
    for i in range(len(X)):
        time_start=time.perf_counter()
        new_X=cov_matrix(np.squeeze(Online_filter.transform(X[index[i]])))
        new_y=y[index[i]]
        result_i=ONC.running(new_X,new_y,'incremental')
        time_end=time.perf_counter()
        if i>=n_train:
            incremental_time_data.append((time_end-time_start)*1000)
            incremental_result_data.append(result_i)
            incremental_accuracy_data.append(accuracy_score(y[index[n_train:i+1]],incremental_result_data)*100)
    all_sub_incremental_time.append(incremental_time_data)
    all_sub_incremental_result.append(incremental_result_data)
    all_sub_incremental_accuracy.append(incremental_accuracy_data)




for k in range(0,9):
    
    plot_rest_time=np.mean(all_sub_rest_x[k],axis=0)
    plot_feet_time=np.mean(all_sub_feet_x[k],axis=0)
    plot_time_x=1/srate*np.arange(0,len(plot_feet_time))
    plt.plot(plot_time_x,plot_rest_time,color='orange',label='Rest')
    plt.plot(plot_time_x,plot_feet_time,color='purple',label='Feet')
    plt.xlim((0,4))
    plt.xlabel('Time/s')
    plt.ylim((-30,50))
    plt.ylabel('Voltage/uV')
    plt.title("Subject {}'s time data".format(k+1))
    plt.legend(loc='upper right')
    plt.savefig('./subject{}_sample_time.jpg'.format(k+1),dpi=1200)
    plt.close()

    """
    plt.specgram(np.mean(all_sub_feet_x[k],axis=0),NFFT=200,Fs=200,noverlap=100)
    plt.xlim((0,4))
    plt.xlabel('Time/s')
    plt.ylim((0,20))
    plt.ylabel('Frequency/Hz')
    plt.title("Subject {}'s PSD data".format(k+1))
    plt.legend(loc='upper right')
    plt.savefig('./subject{}_psd.jpg'.format(k+1),dpi=1200)
    plt.close()
    """
    
    plot_trial_x=np.arange(len(all_sub_static_time[k]))+1
    plot_static_time_y=all_sub_static_time[k]
    plot_retrained_time_y=all_sub_retrained_time[k]
    plot_incremental_time_y=all_sub_incremental_time[k]
    plt.plot(plot_trial_x,plot_static_time_y,color='green',label='Static strategy',linestyle='-')
    plt.plot(plot_trial_x,plot_retrained_time_y,color='blue',label='Retrained strategy',linestyle='-.')
    plt.plot(plot_trial_x,plot_incremental_time_y,color='red',label='Incremental strategy',linestyle=':')
    plt.xticks(np.arange(0,len(plot_trial_x)+1,8),fontsize=8)
    plt.xlim((0,len(plot_trial_x)))
    plt.xlabel('Trials')
    plt.yticks(np.arange(0,251,25),fontsize=8)
    plt.ylim((0,250))
    plt.ylabel('Time/ms')
    plt.grid(linestyle='--')
    plt.title("Code running time of subject {}'s data".format(k+1))
    plt.legend(loc='upper left')
    plt.savefig('./subject{}_running_time.jpg'.format(k+1),dpi=1200)
    plt.close()

    plot_trial_x=np.arange(len(all_sub_static_time[k]))+1
    plot_static_accuracy_y=all_sub_static_accuracy[k]
    plot_retrained_accuracy_y=all_sub_retrained_accuracy[k]
    plot_incremental_accuracy_y=all_sub_incremental_accuracy[k]
    plt.plot(plot_trial_x,plot_static_accuracy_y,color='green',label='Static strategy',linestyle='-')
    plt.plot(plot_trial_x,plot_retrained_accuracy_y,color='blue',label='Retrained strategy',linestyle='--')
    plt.plot(plot_trial_x,plot_incremental_accuracy_y,color='red',label='Incremental strategy',linestyle=':')
    plt.xticks(np.arange(0,len(plot_trial_x)+1,8),fontsize=8)
    plt.xlim((0,len(plot_trial_x)))
    plt.xlabel('Trials')
    plt.yticks(np.arange(0,101,10),fontsize=8)
    plt.ylim((0,100))
    plt.ylabel('Accuracy/%')
    plt.grid(linestyle='--')
    plt.title("Accuracy of subject {}'s data".format(k+1))
    plt.legend(loc='lower right')
    plt.savefig('./subject{}_accuracy.jpg'.format(k+1),dpi=1200)
    plt.close()
    
plot_trial_x=np.arange(len(all_sub_static_time[k]))+1
mean_sub_static_time=np.mean(all_sub_static_time,axis=0)
mean_sub_retrained_time=np.mean(all_sub_retrained_time,axis=0)
mean_sub_incremental_time=np.mean(all_sub_incremental_time,axis=0)
plt.plot(plot_trial_x,mean_sub_static_time,color='green',label='Static strategy',linestyle='-')
plt.plot(plot_trial_x,mean_sub_retrained_time,color='blue',label='Retrained strategy',linestyle='-.')
plt.plot(plot_trial_x,mean_sub_incremental_time,color='red',label='Incremental strategy',linestyle=':')
plt.xticks(np.arange(0,len(plot_trial_x)+1,8),fontsize=8)
plt.xlim((0,len(plot_trial_x)))
plt.xlabel('Trials')
plt.yticks(np.arange(0,251,25),fontsize=8)
plt.ylim((0,250))
plt.ylabel('Time/ms')
plt.grid(linestyle='--')
plt.title("Average code running time of all subject's data")
plt.legend(loc='upper left')
plt.savefig('./subject_average_running.jpg',dpi=1200)
plt.close()

plot_trial_x=np.arange(len(all_sub_static_time[k]))+1
mean_sub_static_accuracy=np.mean(all_sub_static_accuracy,axis=0)
mean_sub_retrained_accuracy=np.mean(all_sub_retrained_accuracy,axis=0)
mean_sub_incremental_accuracy=np.mean(all_sub_incremental_accuracy,axis=0)
plt.plot(plot_trial_x,mean_sub_static_accuracy,color='green',label='Static strategy',linestyle='-')
plt.plot(plot_trial_x,mean_sub_retrained_accuracy,color='blue',label='Retrained strategy',linestyle='-.')
plt.plot(plot_trial_x,mean_sub_incremental_accuracy,color='red',label='Incremental strategy',linestyle=':')
plt.xticks(np.arange(0,len(plot_trial_x)+1,8),fontsize=8)
plt.xlim((0,len(plot_trial_x)))
plt.xlabel('Trials')
plt.yticks(np.arange(0,101,10),fontsize=8)
plt.ylim((0,100))
plt.ylabel('Accuracy/%')
plt.grid(linestyle='--')
plt.title("Average accuracy of all subject data")
plt.legend(loc='lower right')
plt.savefig('./subject_average_accuracy.jpg',dpi=1200)
plt.close()

plot_rest_time=np.mean(np.mean(all_sub_rest_x,axis=0),axis=0)
plot_feet_time=np.mean(np.mean(all_sub_feet_x,axis=0),axis=0)
plot_time_x=1/srate*np.arange(0,len(plot_feet_time))    
plt.plot(plot_time_x,plot_rest_time,color='orange',label='Rest')
plt.plot(plot_time_x,plot_feet_time,color='purple',label='Feet')
plt.xlim((0,4))
plt.xlabel('Time/s')
plt.ylim((-30,50))
plt.ylabel('Voltage/uV')
plt.title("Average sample time data of all subjects")
plt.legend(loc='upper right')
plt.savefig('./subject_average_sample_time.jpg',dpi=1200)
plt.close()

average_static_time=np.round(np.mean(np.array(all_sub_static_time),axis=1),decimals=2)
average_retrained_time=np.round(np.mean(np.array(all_sub_retrained_time),axis=1),decimals=2)
average_incremental_time=np.round(np.mean(np.array(all_sub_incremental_time),axis=1),decimals=2)
final_static_accuracy=np.round(np.array(all_sub_static_accuracy)[:,-1],decimals=2)
final_retrained_accuracy=np.round(np.array(all_sub_retrained_accuracy)[:,-1],decimals=2)
final_incremental_accuracy=np.round(np.array(all_sub_incremental_accuracy)[:,-1],decimals=2)
"""
plt.specgram(np.mean(np.mean(all_sub_rest_x,axis=0),axis=0),NFFT=32,Fs=200,noverlap=8)
plt.xlim((0,4))
plt.xlabel('Time/s')
plt.ylim((0,20))
plt.ylabel('Frequency/Hz')
plt.title("Specgram of average sample time data of all subjects:rest")
plt.savefig('./subject_average_specgram_rest.jpg'.format(k+1),dpi=1200)
plt.close()
"""



