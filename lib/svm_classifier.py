#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:33:11 2019
Updated on Thu Jun 6 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedBaggingClassifier
import multiprocessing
import lib.misc as misc
import os
import time





def grid_search_svm(PARAMS, data_dict):
    pwrs_c = list(np.arange(-5,1,1))
    pwrs_gamma = list(np.arange(-5,1,1))
    C = np.power(2.0,pwrs_c)
    Gamma = np.power(2.0,pwrs_gamma)
    svm_type = 'single'
    njobs = multiprocessing.cpu_count() - 1
    cv_folds = 3
    print('SVM type=', svm_type, ' CV folds=', cv_folds, ' n_jobs=', njobs)

    trainingTimeTaken = 0
    start = time.process_time()
    
    if svm_type == 'single':
        clf = SVC(decision_function_shape='ovo', verbose=0, probability=True)
        tunable_parameters = [{'kernel': ['rbf'], 'gamma': Gamma, 'C': C}]
        CLF_CV = GridSearchCV(clf, tunable_parameters, cv=cv_folds, iid=True, refit=True, n_jobs=njobs, verbose=False)

    elif svm_type == 'bagging':
        clf = SVC(decision_function_shape='ovo', verbose=0, probability=True)
        '''
        This function extracts balanced bootstraps
        '''
        max_features = 1.0
        n_estimators = 10
        bagged_classifier = BalancedBaggingClassifier(base_estimator=clf, sampling_strategy=1.0, n_estimators=n_estimators)
        max_samples = [0.2] #[0.001, 0.005, 0.01, 0.05]
        print('max_samples=', max_samples, ' max_features=', max_features, ' n_estimators=', n_estimators)
        tunable_parameters = {'max_samples' : max_samples, 'base_estimator__gamma': Gamma, 'base_estimator__C': C}
        '''
        Perform Grid search over individual classifiers in the bag
        '''
        CLF_CV = GridSearchCV(bagged_classifier, tunable_parameters, scoring='accuracy', cv=cv_folds, iid=False, refit=True, n_jobs=njobs, verbose=True)

   
    All_train_data = np.append(data_dict['train_data'], data_dict['val_data'], 0)
    All_train_label = np.append(data_dict['train_label'], data_dict['val_label'])

    '''
    Checking if model is already available
    '''
    if not os.path.exists(PARAMS['modelName']):
        CLF_CV.fit(All_train_data, All_train_label)
        model = CLF_CV.best_estimator_
        if PARAMS['save_flag']:
            misc.save_obj(model, PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0])
            misc.save_obj(CLF_CV, PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_All_Models')
    else:
        model = misc.load_obj(PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0])
        CLF_CV = misc.load_obj(PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_All_Models')

    trainingTimeTaken = time.process_time() - start
    
    testingTimeTaken = 0
    start = time.process_time()

    if svm_type=='single':
        optC = str(CLF_CV.best_params_['C'])
        optGamma = str(CLF_CV.best_params_['gamma'])
        countSV = model.n_support_
    elif svm_type=='bagging':
        optC = str(CLF_CV.best_params_['base_estimator__C'])
        optGamma = str(CLF_CV.best_params_['base_estimator__gamma'])
        countSV = [0, 0]
        
    countTrPts = [np.sum(All_train_label==lab) for lab in np.unique(All_train_label)]

    PtdLabels_train = model.predict(All_train_data)
    Predictions_train = model.predict_log_proba(All_train_data)

    PtdLabels_test = model.predict(data_dict['test_data'])
    Predictions_test = model.predict_log_proba(data_dict['test_data'])

    accuracy_train = np.mean(PtdLabels_train.ravel() == All_train_label.ravel()) * 100
    accuracy_test = np.mean(PtdLabels_test.ravel() == data_dict['test_label'].ravel()) * 100
    
    ConfMat_train, fscore_train = misc.getPerformance(PtdLabels_train, All_train_label)
    ConfMat_test, fscore_test = misc.getPerformance(PtdLabels_test, data_dict['test_label'])
    
    Performance_train = np.array([accuracy_train, fscore_train[0], fscore_train[1], fscore_train[2]])
    Performance_test = np.array([accuracy_test, fscore_test[0], fscore_test[1], fscore_test[2]])
    
    print('Accuracy: train=', accuracy_train, ' test=', accuracy_test, 'F-score: train=', fscore_train[-1], ' test=', fscore_test[-1], ' SupportVectors=', countSV)
    testingTimeTaken = time.process_time() - start
    
    
    Train_Params = {
        'model':model,
        'optC': optC,
        'optGamma': optGamma,
        'countSV': countSV,
        'countTrPts': countTrPts,
        'trainingTimeTaken': trainingTimeTaken,
        }
    
    Test_Params = {
        'PtdLabels_train': PtdLabels_train,
        'Predictions_train': Predictions_train,
        'PtdLabels_test': PtdLabels_test,
        'Predictions_test': Predictions_test,
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'Performance_train': Performance_train,
        'Performance_test': Performance_test,
        'testingTimeTaken': testingTimeTaken,
        }

    return Train_Params, Test_Params
