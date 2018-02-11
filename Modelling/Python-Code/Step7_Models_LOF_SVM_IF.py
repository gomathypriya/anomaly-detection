#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:44:55 2018

@author: Gomathypriya Dhanapal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from sklearn.ensemble import IsolationForest


data = pd.read_csv("PY_Thyro.csv") # Read datafile from current folder
target_label = data.columns.values.tolist()[0]  # Extract name of the target column, first column y and other interger columns

nrun = 10  # Number of runs
lof_scores = np.zeros(nrun)
oc_svm_scores = np.zeros(nrun)
isf_scores = np.zeros(nrun)

for i in range(nrun):
    train, test = train_test_split(data, test_size=0.4) # Split data 60% for train and 40% for test
    train.to_csv('train_run'+str(i)+'.csv')  # Save train data for the ith run
    test.to_csv('test_run'+str(i)+'.csv')    # Save test data for the ith run
    y_train = train[target_label]
    x_train = train.drop(target_label, axis=1)

    y_test = test[target_label]
    x_test = test.drop(target_label, axis=1)
    
     # LOF: LocalOutlierFactor
    LOF = LocalOutlierFactor() 
     # LOF is unsupervised and there is no predict function available. So, directly used the test data
    lof_pred = LOF.fit_predict(x_test) # predict return -1 for the outlier
    lof_p = np.zeros(len(y_test))
    lof_p[lof_pred==-1] = 1 # changing the -1 to 1 to be compatitable with the target
    lof_s = roc_auc_score(y_test,lof_p)
    lof_scores[i] = lof_s
    
    # Oneclass SVM
    OC_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    OC_svm.fit(x_train, y = y_train)
    y_pred_test = OC_svm.predict(x_test) # Predict return -1 for the outlier

    y_p_test = np.zeros(len(y_test))
    y_p_test[y_pred_test==-1] = 1 # Changing the -1 to 1 to be compatitable with the target
    test_s = roc_auc_score(y_test,y_p_test)
    oc_svm_scores[i] = test_s
    
    # Isolation Forest
    isf = IsolationForest(max_samples=100)
    isf.fit(x_train, y = y_train)
    isf_pred = isf.predict(x_test) # Predict return -1 for the outlier

    isf_test = np.zeros(len(y_test))
    isf_test[isf_pred == -1] = 1 # Changing the -1 to 1 to be compatitable with the target
    isf_s = roc_auc_score(y_test,isf_test)
    isf_scores[i] = isf_s
    
    predicts = pd.DataFrame({
        'Actual':y_test,
        'LOF':lof_p,
        'SVM':y_p_test,
        'ISF':isf_test
    }) # Save the prediction of 3 methods to predicts dataframe
    predicts.to_csv('predictions_run_'+str(i)+'.csv', sep=',', encoding='utf-8',index=False)
    
final = pd.DataFrame({'LOF':lof_scores, 'One_Class_SVM':oc_svm_scores, 'Isolation_Forest':isf_scores})
final.to_csv('roc_auc_scores.csv', sep=',', encoding='utf-8',index=False)
