# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:39:34 2020

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
from copy import deepcopy

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

TSSVMC_all_FPR_result=[]
TSSVMC_all_ACC_result=[]

TSSVMCSR_all_FPR_result=[]
TSSVMCSR_all_ACC_result=[]

for k in range(1,10):
    

    feet_data_x,feet_data_y,rest_data_x,rest_data_y=get_mi_mat_data(k,limb_channels)
    y=np.concatenate((feet_data_y,rest_data_y))
    raw_X=np.concatenate((feet_data_x,rest_data_x))
    filter_X=np.squeeze(Online_filter.transform(raw_X))
    index=np.arange(len(y))

    TSSVMC=RGC.TSclassifier(clf=svm.SVC(C=4.0))
    TSSVMC_FPR_result=[]
    TSSVMC_ACC_result=[]  
    
    TSSVMCSR=RGC.TSclassifier(clf=svm.SVC(C=4.0))
    TSSVMCSR_FPR_result=[]
    TSSVMCSR_ACC_result=[]   

    kf=KFold(n_splits=5, shuffle=True, random_state=None)
    
    for train_index,test_index in kf.split(filter_X):
        xtrain=filter_X[train_index]
        ytrain=y[train_index]
        rest_index=np.squeeze(np.array(np.where(ytrain==0)))
        foot_index=np.squeeze(np.array(np.where(ytrain==1)))
        rest_data=xtrain[rest_index]
        foot_data=xtrain[foot_index]
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

        TSSVMC_X=cov_matrix(filter_X).copy()
        TSSVMC.fit(TSSVMC_X[train_index],y[train_index])
        TSSVMC_predict=TSSVMC.predict(TSSVMC_X[test_index])
        TSSVMC_Cmat=confusion_matrix(y[test_index],TSSVMC_predict)
        TSSVMC_FPR=TSSVMC_Cmat[0][1]/(TSSVMC_Cmat[0][1]+TSSVMC_Cmat[0][0])
        TSSVMC_FPR_result.append(TSSVMC_FPR*100)
        TSSVMC_ACC=accuracy_score(y[test_index],TSSVMC_predict)
        TSSVMC_ACC_result.append(TSSVMC_ACC*100)

        
        TSSVMCSR_X=cov_matrix(filter_X[:,srca_chan_index,:]).copy()
        TSSVMCSR.fit(TSSVMCSR_X[train_index],y[train_index])
        TSSVMCSR_predict=TSSVMCSR.predict(TSSVMCSR_X[test_index])
        TSSVMCSR_Cmat=confusion_matrix(y[test_index],TSSVMCSR_predict)
        TSSVMCSR_FPR=TSSVMCSR_Cmat[0][1]/(TSSVMCSR_Cmat[0][1]+TSSVMCSR_Cmat[0][0])
        TSSVMCSR_FPR_result.append(TSSVMCSR_FPR*100)
        TSSVMCSR_ACC=accuracy_score(y[test_index],TSSVMCSR_predict)
        TSSVMCSR_ACC_result.append(TSSVMCSR_ACC*100)

    TSSVMC_all_FPR_result.append(TSSVMC_FPR_result)   
    TSSVMC_all_ACC_result.append(TSSVMC_ACC_result)

    TSSVMCSR_all_FPR_result.append(TSSVMCSR_FPR_result)   
    TSSVMCSR_all_ACC_result.append(TSSVMCSR_ACC_result)

mean_TSSVMC_all_FPR_result=np.mean(TSSVMC_all_FPR_result,axis=1)
std_TSSVMC_all_FPR_result=np.std(TSSVMC_all_FPR_result,axis=1)
mean_TSSVMCSR_all_FPR_result=np.mean(TSSVMCSR_all_FPR_result,axis=1)
std_TSSVMCSR_all_FPR_result=np.std(TSSVMCSR_all_FPR_result,axis=1)    
fin_FPR_result=np.around(np.array((mean_TSSVMC_all_FPR_result,mean_TSSVMCSR_all_FPR_result)),decimals=2)   

mean_TSSVMC_all_ACC_result=np.mean(TSSVMC_all_ACC_result,axis=1)
std_TSSVMC_all_ACC_result=np.std(TSSVMC_all_ACC_result,axis=1)
mean_TSSVMCSR_all_ACC_result=np.mean(TSSVMCSR_all_ACC_result,axis=1)
std_TSSVMCSR_all_ACC_result=np.std(TSSVMCSR_all_ACC_result,axis=1)    
fin_ACC_result=np.around(np.array((mean_TSSVMC_all_ACC_result,mean_TSSVMCSR_all_ACC_result)),decimals=2)   

sub_num=np.arange(1,10)
width_val = 0.2
plt.style.use('ggplot')
plt.title("Accuracy of all subject's data without or with SRCA", fontsize=12)
plt.xticks(sub_num)
plt.xlim((0,len(sub_num)+1))
plt.xlabel('Subjects')
plt.yticks(np.arange(50,100,5),fontsize=8)
plt.ylim((50,100))
plt.ylabel('Accuracy/%')
plt.grid(axis='x')
plt.bar(sub_num-width_val,mean_TSSVMC_all_ACC_result,width=width_val,label='without SRCA')
plt.bar(sub_num+width_val,mean_TSSVMCSR_all_ACC_result,width=width_val,label='with SRCA')
plt.legend(loc='lower right')
plt.savefig('ACC without or with SRCA.jpg'.format(k+1),dpi=1200)
plt.close()


sub_num=np.arange(1,10)
width_val = 0.2
plt.style.use('ggplot')
plt.title("False positive rate of all subject's data without or with SRCA", fontsize=12)
plt.xticks(sub_num)
plt.xlim((0,len(sub_num)+1))
plt.xlabel('Subjects')
plt.yticks(np.arange(0,50,5),fontsize=8)
plt.ylim((0,50))
plt.ylabel('False positive rate/%')
plt.grid(axis='x')
plt.bar(sub_num-width_val,mean_TSSVMC_all_FPR_result,width=width_val,label='without SRCA')
plt.bar(sub_num+width_val,mean_TSSVMCSR_all_FPR_result,width=width_val,label='with SRCA')
plt.legend(loc='upper right')
plt.savefig('FPR without or with SRCA.jpg'.format(k+1),dpi=1200)
plt.close()


"""
TSSVMC0_all_FPR_result=[]
TSSVMC1_all_FPR_result=[]
TSSVMC2_all_FPR_result=[]
TSSVMC3_all_FPR_result=[]
TSSVMC4_all_FPR_result=[]

TSSVMC0_all_ACC_result=[]
TSSVMC1_all_ACC_result=[]
TSSVMC2_all_ACC_result=[]
TSSVMC3_all_ACC_result=[]
TSSVMC4_all_ACC_result=[]



for k in range(1,10):
    feet_data_x,feet_data_y,rest_data_x,rest_data_y=get_mi_mat_data(k,limb_channels)
    y=np.concatenate((feet_data_y,rest_data_y))
    raw_X=np.concatenate((feet_data_x,rest_data_x))
    f_X=np.squeeze(Online_filter.transform(raw_X))
    index=np.arange(len(y))

    X_mat=cov_matrix(f_X)

    
    TSSVMC0_X=X_mat.copy()
    TSSVMC1_X=X_mat.copy()
    TSSVMC2_X=X_mat.copy()
    TSSVMC3_X=X_mat.copy()
    TSSVMC4_X=X_mat.copy()
    
    TSSVMC0=RGC.TSclassifier(clf=svm.SVC(C=1.0))
    TSSVMC0_FPR_result=[]
    TSSVMC0_ACC_result=[]
    TSSVMC1=RGC.TSclassifier(clf=svm.SVC(C=2.0))
    TSSVMC1_FPR_result=[]
    TSSVMC1_ACC_result=[]
    TSSVMC2=RGC.TSclassifier(clf=svm.SVC(C=4.0))
    TSSVMC2_FPR_result=[]
    TSSVMC2_ACC_result=[]
    TSSVMC3=RGC.TSclassifier(clf=svm.SVC(C=8.0))
    TSSVMC3_FPR_result=[]
    TSSVMC3_ACC_result=[]
    TSSVMC4=RGC.TSclassifier(clf=svm.SVC(C=16.0))
    TSSVMC4_FPR_result=[]
    TSSVMC4_ACC_result=[]
    
    kf=KFold(n_splits=5, shuffle=True, random_state=None)
    
    for train_index,test_index in kf.split(X_mat):
      
        TSSVMC0.fit(TSSVMC0_X[train_index],y[train_index])
        TSSVMC0_predict=TSSVMC0.predict(TSSVMC0_X[test_index])
        
        TSSVMC0_Cmat=confusion_matrix(y[test_index],TSSVMC0_predict)
        TSSVMC0_FPR=TSSVMC0_Cmat[0][1]/(TSSVMC0_Cmat[0][1]+TSSVMC0_Cmat[0][0])
        TSSVMC0_FPR_result.append(TSSVMC0_FPR*100)
        TSSVMC0_ACC=accuracy_score(y[test_index],TSSVMC0_predict)
        TSSVMC0_ACC_result.append(TSSVMC0_ACC*100)

        
        
        TSSVMC1.fit(TSSVMC1_X[train_index],y[train_index])
        TSSVMC1_predict=TSSVMC1.predict(TSSVMC1_X[test_index])
        
        TSSVMC1_Cmat=confusion_matrix(y[test_index],TSSVMC1_predict)
        TSSVMC1_FPR=TSSVMC1_Cmat[0][1]/(TSSVMC1_Cmat[0][1]+TSSVMC1_Cmat[0][0])
        TSSVMC1_FPR_result.append(TSSVMC1_FPR*100)
        TSSVMC1_ACC=accuracy_score(y[test_index],TSSVMC1_predict)
        TSSVMC1_ACC_result.append(TSSVMC1_ACC*100)
                
        TSSVMC2.fit(TSSVMC2_X[train_index],y[train_index])
        TSSVMC2_predict=TSSVMC2.predict(TSSVMC2_X[test_index])
        
        TSSVMC2_Cmat=confusion_matrix(y[test_index],TSSVMC2_predict)
        TSSVMC2_FPR=TSSVMC2_Cmat[0][1]/(TSSVMC2_Cmat[0][1]+TSSVMC2_Cmat[0][0])
        TSSVMC2_FPR_result.append(TSSVMC2_FPR*100)
        TSSVMC2_ACC=accuracy_score(y[test_index],TSSVMC2_predict)
        TSSVMC2_ACC_result.append(TSSVMC2_ACC*100)
                
        TSSVMC3.fit(TSSVMC3_X[train_index],y[train_index])
        TSSVMC3_predict=TSSVMC3.predict(TSSVMC3_X[test_index])
        
        TSSVMC3_Cmat=confusion_matrix(y[test_index],TSSVMC3_predict)
        TSSVMC3_FPR=TSSVMC3_Cmat[0][1]/(TSSVMC3_Cmat[0][1]+TSSVMC3_Cmat[0][0])
        TSSVMC3_FPR_result.append(TSSVMC3_FPR*100)
        TSSVMC3_ACC=accuracy_score(y[test_index],TSSVMC3_predict)
        TSSVMC3_ACC_result.append(TSSVMC3_ACC*100)

        TSSVMC3.fit(TSSVMC3_X[train_index],y[train_index])
        TSSVMC3_predict=TSSVMC3.predict(TSSVMC3_X[test_index])
        
        TSSVMC3_Cmat=confusion_matrix(y[test_index],TSSVMC3_predict)
        TSSVMC3_FPR=TSSVMC3_Cmat[0][1]/(TSSVMC3_Cmat[0][1]+TSSVMC3_Cmat[0][0])
        TSSVMC3_FPR_result.append(TSSVMC3_FPR*100)
        TSSVMC3_ACC=accuracy_score(y[test_index],TSSVMC3_predict)
        TSSVMC3_ACC_result.append(TSSVMC3_ACC*100)

        TSSVMC4.fit(TSSVMC4_X[train_index],y[train_index])
        TSSVMC4_predict=TSSVMC4.predict(TSSVMC4_X[test_index])
        
        TSSVMC4_Cmat=confusion_matrix(y[test_index],TSSVMC4_predict)
        TSSVMC4_FPR=TSSVMC4_Cmat[0][1]/(TSSVMC4_Cmat[0][1]+TSSVMC4_Cmat[0][0])
        TSSVMC4_FPR_result.append(TSSVMC4_FPR*100)
        TSSVMC4_ACC=accuracy_score(y[test_index],TSSVMC4_predict)
        TSSVMC4_ACC_result.append(TSSVMC4_ACC*100)


    TSSVMC0_all_FPR_result.append(TSSVMC0_FPR_result)
    TSSVMC1_all_FPR_result.append(TSSVMC1_FPR_result)
    TSSVMC2_all_FPR_result.append(TSSVMC2_FPR_result)
    TSSVMC3_all_FPR_result.append(TSSVMC3_FPR_result)
    TSSVMC4_all_FPR_result.append(TSSVMC4_FPR_result)
    
    TSSVMC0_all_ACC_result.append(TSSVMC0_ACC_result)
    TSSVMC1_all_ACC_result.append(TSSVMC1_ACC_result)
    TSSVMC2_all_ACC_result.append(TSSVMC2_ACC_result)
    TSSVMC3_all_ACC_result.append(TSSVMC3_ACC_result)
    TSSVMC4_all_ACC_result.append(TSSVMC4_ACC_result)
  
mean_TSSVMC0_all_FPR_result=np.mean(TSSVMC0_all_FPR_result,axis=1)
std_TSSVMC0_all_FPR_result=np.std(TSSVMC0_all_FPR_result,axis=1)
mean_TSSVMC1_all_FPR_result=np.mean(TSSVMC1_all_FPR_result,axis=1)
std_TSSVMC1_all_FPR_result=np.std(TSSVMC1_all_FPR_result,axis=1)
mean_TSSVMC2_all_FPR_result=np.mean(TSSVMC2_all_FPR_result,axis=1)
std_TSSVMC2_all_FPR_result=np.std(TSSVMC2_all_FPR_result,axis=1)
mean_TSSVMC3_all_FPR_result=np.mean(TSSVMC3_all_FPR_result,axis=1)
std_TSSVMC3_all_FPR_result=np.std(TSSVMC3_all_FPR_result,axis=1)
mean_TSSVMC4_all_FPR_result=np.mean(TSSVMC4_all_FPR_result,axis=1)
std_TSSVMC4_all_FPR_result=np.std(TSSVMC4_all_FPR_result,axis=1)

fin_FPR_result=np.around(np.array((mean_TSSVMC0_all_FPR_result,mean_TSSVMC1_all_FPR_result,mean_TSSVMC2_all_FPR_result,
                         mean_TSSVMC3_all_FPR_result,mean_TSSVMC4_all_FPR_result)),decimals=2)



mean_TSSVMC0_all_ACC_result=np.mean(TSSVMC0_all_ACC_result,axis=1)
std_TSSVMC0_all_ACC_result=np.std(TSSVMC0_all_ACC_result,axis=1)
mean_TSSVMC1_all_ACC_result=np.mean(TSSVMC1_all_ACC_result,axis=1)
std_TSSVMC1_all_ACC_result=np.std(TSSVMC1_all_ACC_result,axis=1)
mean_TSSVMC2_all_ACC_result=np.mean(TSSVMC2_all_ACC_result,axis=1)
std_TSSVMC2_all_ACC_result=np.std(TSSVMC2_all_ACC_result,axis=1)
mean_TSSVMC3_all_ACC_result=np.mean(TSSVMC3_all_ACC_result,axis=1)
std_TSSVMC3_all_ACC_result=np.std(TSSVMC3_all_ACC_result,axis=1)
mean_TSSVMC4_all_ACC_result=np.mean(TSSVMC4_all_ACC_result,axis=1)
std_TSSVMC4_all_ACC_result=np.std(TSSVMC4_all_ACC_result,axis=1)

fin_ACC_result=np.around(np.array((mean_TSSVMC0_all_ACC_result,mean_TSSVMC1_all_ACC_result,mean_TSSVMC2_all_ACC_result,
                         mean_TSSVMC3_all_ACC_result,mean_TSSVMC4_all_ACC_result)),decimals=2)

sub_num=np.arange(1,10)
width_val = 0.6
plt.style.use('ggplot')
plt.title("False positive rate of all subject's data with TS+SVM, C=1",fontsize=12)
plt.xticks(sub_num)
plt.xlim((0,len(sub_num)+1))
plt.xlabel('Subjects')
plt.yticks(np.arange(0,25,1),fontsize=8)
plt.ylim((0,25))
plt.ylabel('False positive rate/%')
plt.grid(axis='x')
plt.bar(sub_num-0*width_val,mean_TSSVMC1_all_FPR_result,width=width_val)
plt.legend(loc='upper right')
plt.savefig('FPR, C=1.jpg'.format(k+1),dpi=1200)
plt.close()

sub_num=np.arange(1,10)
width_val = 0.15
plt.style.use('ggplot')
plt.title("False positive rate of all subject's data with TS+SVM and different C", fontsize=12)
plt.xticks(sub_num)
plt.xlim((0,len(sub_num)+1))
plt.xlabel('Subjects')
plt.yticks(np.arange(0,25,1),fontsize=8)
plt.ylim((0,25))
plt.ylabel('False positive rate/%')
plt.grid(axis='x')
plt.bar(sub_num-2*width_val,mean_TSSVMC0_all_FPR_result,width=width_val,label='C=0.5')
plt.bar(sub_num-1*width_val,mean_TSSVMC1_all_FPR_result,width=width_val,label='C=1.0')
plt.bar(sub_num+0*width_val,mean_TSSVMC2_all_FPR_result,width=width_val,label='C=2.0')
plt.bar(sub_num+1*width_val,mean_TSSVMC3_all_FPR_result,width=width_val,label='C=4.0')
plt.bar(sub_num+2*width_val,mean_TSSVMC4_all_FPR_result,width=width_val,label='C=8.0')
plt.legend(loc='upper right')
plt.savefig('FPR with different C.jpg'.format(k+1),dpi=1200)
plt.close()

sub_num=np.arange(1,10)
width_val = 0.6
plt.style.use('ggplot')
plt.title("Accuracy of all subject's data with TS+SVM, C=1",fontsize=12)
plt.xticks(sub_num)
plt.xlim((0,len(sub_num)+1))
plt.xlabel('Subjects')
plt.yticks(np.arange(80,100,1),fontsize=8)
plt.ylim((80,100))
plt.ylabel('Accuracy/%')
plt.grid(axis='x')
plt.bar(sub_num-0*width_val,mean_TSSVMC1_all_ACC_result,width=width_val)
plt.legend(loc='lower right')
plt.savefig('ACC, C=1.jpg'.format(k+1),dpi=1200)
plt.close()

sub_num=np.arange(1,10)
width_val = 0.15
plt.style.use('ggplot')
plt.title("Accuracy of all subject's data with TS+SVM and different C", fontsize=12)
plt.xticks(sub_num)
plt.xlim((0,len(sub_num)+1))
plt.xlabel('Subjects')
plt.yticks(np.arange(80,100,1),fontsize=8)
plt.ylim((80,100))
plt.ylabel('Accuracy/%')
plt.grid(axis='x')
plt.bar(sub_num-2*width_val,mean_TSSVMC0_all_ACC_result,width=width_val,label='C=0.5')
plt.bar(sub_num-1*width_val,mean_TSSVMC1_all_ACC_result,width=width_val,label='C=1.0')
plt.bar(sub_num+0*width_val,mean_TSSVMC2_all_ACC_result,width=width_val,label='C=2.0')
plt.bar(sub_num+1*width_val,mean_TSSVMC3_all_ACC_result,width=width_val,label='C=4.0')
plt.bar(sub_num+2*width_val,mean_TSSVMC4_all_ACC_result,width=width_val,label='C=8.0')
plt.legend(loc='lower right')
plt.savefig('ACC with different C.jpg'.format(k+1),dpi=1200)
plt.close()
"""