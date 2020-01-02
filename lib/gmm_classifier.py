#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:03:36 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import numpy as np
from sklearn.mixture import GaussianMixture
import lib.misc as misc
import os




def grid_search_gmm(PARAMS, data_dict):
    K = 10
    gmmModel_mu = GaussianMixture(n_components=K, max_iter = 1000)
    gmmModel_sp = GaussianMixture(n_components=K, max_iter = 1000)

    All_train_data = np.append(data_dict['train_data'], data_dict['val_data'], 0)
    All_train_label = np.append(data_dict['train_label'], data_dict['val_label'])
    
    mu_idx = np.squeeze(np.where(All_train_label==0))
    sp_idx = np.squeeze(np.where(All_train_label==1))

    '''
    Checking if model is already available
    '''
    gmmModelFileName = PARAMS['opDir'] + PARAMS['modelName'].split('/')[-1].split('.')[0]+'_gmmModel_mu_K=' + str(K) + '.pkl'
    if not os.path.exists(gmmModelFileName):
        gmmModel_mu.fit(All_train_data[mu_idx, :])
        gmmModel_sp.fit(All_train_data[sp_idx, :])
        
        if PARAMS['save_flag']:
            misc.save_obj(gmmModel_mu, PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_gmmModel_mu_K=' + str(K))
            misc.save_obj(gmmModel_sp, PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_gmmModel_sp_K=' + str(K))
    else:
        gmmModel_mu = misc.load_obj(PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_gmmModel_mu_K=' + str(K))
        gmmModel_sp = misc.load_obj(PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_gmmModel_sp_K=' + str(K))

    score_train_mu = np.array(gmmModel_mu.score_samples(All_train_data), ndmin=2).T
    score_train_sp = np.array(gmmModel_sp.score_samples(All_train_data), ndmin=2).T
    print('scores shape: ', np.shape(score_train_mu), np.shape(score_train_sp))
    score_train = np.append(score_train_mu, score_train_sp, 1)
    print('score_train: ', np.shape(score_train))
    PtdLabels_train = np.argmax(score_train, axis=1)

    score_test_mu = np.array(gmmModel_mu.score_samples(data_dict['test_data']), ndmin=2).T
    score_test_sp = np.array(gmmModel_sp.score_samples(data_dict['test_data']), ndmin=2).T
    score_test = np.append(score_test_mu, score_test_sp, 1)
    print('score_test: ', np.shape(score_test))
    PtdLabels_test = np.argmax(score_test, axis=1)
    
    accuracy_train = np.mean(PtdLabels_train.ravel() == All_train_label.ravel()) * 100
    accuracy_test = np.mean(PtdLabels_test.ravel() == data_dict['test_label'].ravel()) * 100
    
    ConfMat_train, fscore_train = misc.getPerformance(PtdLabels_train, All_train_label)
    ConfMat_test, fscore_test = misc.getPerformance(PtdLabels_test, data_dict['test_label'])
    
    Performance_train = np.array([accuracy_train, fscore_train[0], fscore_train[1], fscore_train[2]])
    Performance_test = np.array([accuracy_test, fscore_test[0], fscore_test[1], fscore_test[2]])
    
    print('Accuracy: train=', accuracy_train, ' test=', accuracy_test, 'F-score: train=', fscore_train[-1], ' test=', fscore_test[-1])

    return score_test, PtdLabels_test, K, Performance_train, Performance_test
