#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:01:03 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import configparser
import lib.misc as misc
from sklearn import mixture
import pickle



'''
This function learns GMMs by taking all data of any i^th sequence at a time. 
Finally, k GMMs are learned, where k is the number of sequences in the the SPS matrix
''' 
def learn_seq_gmm(PARAMS, seq_data, optFileGMM):
    flattened_seq_data = np.transpose(np.array(seq_data.flatten(), ndmin=2))
    gmm = mixture.GaussianMixture(n_components=PARAMS['numMix'], covariance_type='full')

    gmm.fit(flattened_seq_data)
    if PARAMS['save_flag']:
        print('Saving GMM model as ', optFileGMM)
        with open(optFileGMM, 'wb') as file:
            pickle.dump(gmm, file)



'''
This function computes posterior probability features
''' 
def get_gmm_posterior_features(PARAMS, seqNum, seq_data, file_mark, data_type):
    data_lkhd = np.empty(shape=(np.size(file_mark),len(PARAMS['classes'])*PARAMS['numMix']))
    
    '''
    Generating the likelihood features from the learned GMMs
    '''
    print('Generating likelihood features')
    file_mark_start = 0
    seq_data = np.array(seq_data, ndmin=2)
    print('Shape of seq_data: ', np.shape(seq_data), np.shape(data_lkhd))
    
    for i in range(len(file_mark)):
        flattened_test_data = np.transpose(np.array(seq_data[0,file_mark_start:file_mark[i]], ndmin=2))
        if not np.size(flattened_test_data)>1:
            continue
    
        for clNum in range(len(PARAMS['classes'])):
            gmm = None
            '''
            Load already learned GMM
            '''
            optFileGMM = PARAMS['gmmPath'] + '/Cl' + str(clNum) + '_seq' + str(seqNum) + '_train_data_gmm.pkl'
            with open(optFileGMM, 'rb') as file:  
                gmm = pickle.load(file)

            proba = gmm.predict_proba(flattened_test_data)
            mean_idx = np.ndarray.argsort(np.squeeze(gmm.means_))
            proba = proba[:, mean_idx]
            proba_fv = np.mean(proba, axis=0)
    
            data_lkhd[i,clNum*PARAMS['numMix']:(clNum+1)*PARAMS['numMix']] = proba_fv
        
        file_mark_start = file_mark[i]
        
    print('Likelihood features computed')

    return data_lkhd






def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['compute_CBoW_features.py']
    PARAMS = {
            'folder': section['folder'],
            'numMix': int(section['numMix']),
            'num_seq': int(section['num_seq']),
            'save_flag': section.getboolean('save_flag'),
            'TS_PART': float(section['TS_PART']),
            'classes':{},
            'iterations': int(section['iterations']),
            'feat_type': section['feat_type'],
            }

    return PARAMS





if __name__ == '__main__':

    PARAMS = __init__()

    dirpath, dirnames, filenames = next(os.walk(PARAMS['folder']))
    clCount = 0
    for i in range(len(dirnames)):
        if dirnames[i].startswith('__') or dirnames[i].startswith('.'):
            continue
        PARAMS['classes'].setdefault(clCount,[]).append(dirnames[i])
        clCount += 1
    
    print(PARAMS['classes'])
    
    Cl = np.array(list(range(len(PARAMS['classes']))))
    
    print('Train folder: ',PARAMS['folder'])


    '''
    This function generates CBoW feature from the Peak trace matrices of the classes
    by learning SEPARATE GMMs and extracting posterior probabilities from them
    '''
    
    for numIter in range(PARAMS['iterations']):
        PARAMS['opDir'] = PARAMS['folder'] + '/' + PARAMS['feat_type'] + '_mix'+ str(PARAMS['numMix']) + '_iter' + str(numIter) + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
            
        file_list_fName = PARAMS['folder'] + '/file_list_iter' + str(numIter) + '.pkl'
        if not os.path.exists(file_list_fName):
            FL_Ret = misc.get_file_list(PARAMS)
            misc.save_obj(FL_Ret, PARAMS['folder'], 'file_list_iter' + str(numIter))
            # This is saved in the feature folder of specific iteration for loading the correct training and testing files during classification
            misc.save_obj(FL_Ret, PARAMS['opDir'], 'file_list_iter' + str(numIter))
        else:
            FL_Ret = misc.load_obj(PARAMS['folder'], 'file_list_iter' + str(numIter))
            # This is saved in the feature folder of specific iteration for loading the correct training and testing files during classification
            misc.save_obj(FL_Ret, PARAMS['opDir'], 'file_list_iter' + str(numIter))
    
    
        PARAMS['tempFolder'] = PARAMS['folder'] + '/__temp/iter' + str(numIter) + '/'
        if not os.path.exists(PARAMS['tempFolder']):
            os.makedirs(PARAMS['tempFolder'])
        PARAMS['gmmPath'] = PARAMS['folder'] + '/__GMMs/iter' + str(numIter) + '/'
        if not os.path.exists(PARAMS['gmmPath']):
            os.makedirs(PARAMS['gmmPath'])

        for clNum in range(len(PARAMS['classes'])):
            fold = PARAMS['classes'][clNum][0]
            data_path = PARAMS['folder'] + '/' + fold 	#path where the peak trace matrices are
            print('Computing features for ', PARAMS['classes'][clNum][0])
            
            train_files = FL_Ret['file_list_train'][PARAMS['classes'][clNum][0]][0]
            val_files = FL_Ret['file_list_val'][PARAMS['classes'][clNum][0]][0]
            test_files = FL_Ret['file_list_test'][PARAMS['classes'][clNum][0]][0]

            seqCount = 0
            for seqNum in range(PARAMS['num_seq']):
                '''
                Reading in the files and storing each sequence as a separate feature 
                vector appended with its corresponding file mark information
                '''
                print('\n.................................................... Seq:', seqNum, ' : ', PARAMS['classes'][clNum][0])
                
                tempFile = PARAMS['tempFolder'] + '/train_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    tr_data_seq, tr_file_mark, tr_data_file_mark = misc.load_files_cbow_features(data_path, train_files, seqNum)
                    np.savez(tempFile, data_seq=tr_data_seq, file_mark=tr_file_mark, data_file_mark=tr_data_file_mark)
                else:
                    tr_data_seq = np.load(tempFile)['data_seq']
                    tr_file_mark = np.load(tempFile)['file_mark']
                    tr_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Train seq data loaded')
                print('                  Training files read ', np.shape(tr_data_seq))
    
                tempFile = PARAMS['tempFolder'] + '/val_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    v_data_seq, v_file_mark, v_data_file_mark = misc.load_files_cbow_features(data_path, val_files, seqNum)
                    np.savez(tempFile, data_seq=v_data_seq, file_mark=v_file_mark, data_file_mark=v_data_file_mark)
                else:
                    v_data_seq = np.load(tempFile)['data_seq']
                    v_file_mark = np.load(tempFile)['file_mark']
                    v_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Validation seq data loaded')
                print('                  Validation files read ', np.shape(v_data_seq))
    
                tempFile = PARAMS['tempFolder'] + '/test_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    ts_data_seq, ts_file_mark, ts_data_file_mark = misc.load_files_cbow_features(data_path, test_files, seqNum)
                    np.savez(tempFile, data_seq=ts_data_seq, file_mark=ts_file_mark, data_file_mark=ts_data_file_mark)
                else:
                    ts_data_seq = np.load(tempFile)['data_seq']
                    ts_file_mark = np.load(tempFile)['file_mark']
                    ts_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Test seq data loaded')
                print('                  Testing files read ', np.shape(ts_data_seq))
                
                print('Data shape: ',np.shape(tr_data_seq), np.shape(v_data_seq), np.shape(ts_data_seq))
                # -----------------------------------------------------------------
                
                optFileGMM = PARAMS['gmmPath'] + 'Cl' + str(clNum) + '_seq' + str(seqNum) + '_train_data_gmm.pkl'
#                learn_seq_gmm(PARAMS, tr_data_seq, optFileGMM)

                '''
                Combining training and validation sequences to learn the GMMs
                '''
                gmm_training_data_seq = tr_data_seq
                gmm_training_data_seq = np.append(gmm_training_data_seq, v_data_seq)
                learn_seq_gmm(PARAMS, gmm_training_data_seq, optFileGMM)


        
        for clNum in range(len(PARAMS['classes'])):
            fold = PARAMS['classes'][clNum][0]
            data_path = PARAMS['folder'] + '/' + fold 	#path where the Peak trace matrices are
            print('Computing features for ', PARAMS['classes'][clNum][0])
            
            tr_data_lkhd = np.empty([])
            v_data_lkhd = np.empty([])
            ts_data_lkhd = np.empty([])

            tr_data_lkhd_merged = np.empty([])
            v_data_lkhd_merged = np.empty([])
            ts_data_lkhd_merged = np.empty([])
            
            train_files = FL_Ret['file_list_train'][PARAMS['classes'][clNum][0]][0]
            val_files = FL_Ret['file_list_val'][PARAMS['classes'][clNum][0]][0]
            test_files = FL_Ret['file_list_test'][PARAMS['classes'][clNum][0]][0]
            
            seqCount = 0
            for seqNum in range(PARAMS['num_seq']):
                '''
                Reading in the files and storing each sequence as a separate feature 
                vector appended with its corresponding file mark information
                '''
                print('\n.................................................... Seq:', seqNum, ' : ', PARAMS['classes'][clNum][0])
                
                tempFile = PARAMS['tempFolder'] + '/train_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    tr_data_seq, tr_file_mark, tr_data_file_mark = misc.load_files_cbow_features(data_path, train_files, seqNum)
                    np.savez(tempFile, data_seq=tr_data_seq, file_mark=tr_file_mark, data_file_mark=tr_data_file_mark)
                else:
                    tr_data_seq = np.load(tempFile)['data_seq']
                    tr_file_mark = np.load(tempFile)['file_mark']
                    tr_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Train seq data loaded')
                print('                  Training files read ', np.shape(tr_data_seq))
    
                tempFile = PARAMS['tempFolder'] + '/val_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    v_data_seq, v_file_mark, v_data_file_mark = misc.load_files_cbow_features(data_path, val_files, seqNum)
                    np.savez(tempFile, data_seq=v_data_seq, file_mark=v_file_mark, data_file_mark=v_data_file_mark)
                else:
                    v_data_seq = np.load(tempFile)['data_seq']
                    v_file_mark = np.load(tempFile)['file_mark']
                    v_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Validation seq data loaded')
                print('                  Validation files read ', np.shape(v_data_seq))
    
                tempFile = PARAMS['tempFolder'] + '/test_cl' + str(clNum) + '_seq' + str(seqNum) + '.npz'
                if not os.path.exists(tempFile):
                    ts_data_seq, ts_file_mark, ts_data_file_mark = misc.load_files_cbow_features(data_path, test_files, seqNum)
                    np.savez(tempFile, data_seq=ts_data_seq, file_mark=ts_file_mark, data_file_mark=ts_data_file_mark)
                else:
                    ts_data_seq = np.load(tempFile)['data_seq']
                    ts_file_mark = np.load(tempFile)['file_mark']
                    ts_data_file_mark = np.load(tempFile, allow_pickle=True)['data_file_mark']
                    #print('                  Test seq data loaded')
                print('                  Testing files read ', np.shape(ts_data_seq))
                
                print('Data shape: ',np.shape(tr_data_seq), np.shape(v_data_seq), np.shape(ts_data_seq))
                # -----------------------------------------------------------------
                
                '''
                Generating GMM likelihood features for each sequence
                '''
                print('Computing GMM likelihood features ...')
                tr_data_lkhd = get_gmm_posterior_features(PARAMS, seqNum, tr_data_seq, tr_file_mark, 'train_data')
                print('                                     Training set done ', np.shape(tr_data_lkhd))
    
                v_data_lkhd = get_gmm_posterior_features(PARAMS, seqNum, v_data_seq, v_file_mark, 'val_data')
                print('                                     Validation set done ', np.shape(v_data_lkhd))
    
                ts_data_lkhd = get_gmm_posterior_features(PARAMS, seqNum, ts_data_seq, ts_file_mark, 'test_data')
                print('                                     Testing set done ', np.shape(ts_data_lkhd))
                # -----------------------------------------------------------------
                
                
                if np.size(tr_data_lkhd_merged)<=1:
                    tr_data_lkhd_merged = tr_data_lkhd
                    v_data_lkhd_merged = v_data_lkhd
                    ts_data_lkhd_merged = ts_data_lkhd
                else:
                    tr_data_lkhd_merged = np.append(tr_data_lkhd_merged, tr_data_lkhd, 1)
                    v_data_lkhd_merged = np.append(v_data_lkhd_merged, v_data_lkhd, 1)
                    ts_data_lkhd_merged = np.append(ts_data_lkhd_merged, ts_data_lkhd, 1)
                    
                seqCount += 1
                if np.sum(np.isnan(tr_data_lkhd))>0:
                    print('\n\n\nNaNs seqNum=',seqNum,'\n\n\n')
                    
            
            '''
            To save the CBoW interval features file wise so that they can be merged later on
            '''
            opDirIter = PARAMS['opDir'] + '/' + fold + '/'
            if not os.path.exists(opDirIter):
                os.makedirs(opDirIter)
            # Dict gets stored as a numpy.ndarray object, so .item() is required
            tr_data_file_mark = tr_data_file_mark.item()
            tr_data_file_mark_keys = [x for x in tr_data_file_mark.keys()]
            tr_file_data = np.empty([])
            for i in range(len(tr_data_file_mark_keys)):
                featFile = opDirIter + tr_data_file_mark_keys[i]
                idx = tr_data_file_mark[tr_data_file_mark_keys[i]][0]
                fileData = tr_data_lkhd_merged[idx[0]:idx[1],:]
                np.save(featFile, fileData)

            v_data_file_mark = v_data_file_mark.item()
            v_data_file_mark_keys = [x for x in v_data_file_mark.keys()]
            for i in range(len(v_data_file_mark_keys)):
                featFile = opDirIter + v_data_file_mark_keys[i]
                idx = v_data_file_mark[v_data_file_mark_keys[i]][0]
                fileData = v_data_lkhd_merged[idx[0]:idx[1],:]
                np.save(featFile, fileData)

            ts_data_file_mark = ts_data_file_mark.item()
            ts_data_file_mark_keys = [x for x in ts_data_file_mark.keys()]
            for i in range(len(ts_data_file_mark_keys)):
                featFile = opDirIter + ts_data_file_mark_keys[i]
                idx = ts_data_file_mark[ts_data_file_mark_keys[i]][0]
                fileData = ts_data_lkhd_merged[idx[0]:idx[1],:]
                np.save(featFile, fileData)
            '''
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            '''
        print('GMM likelihood features saved.')


    print('\n\n\nGenerated GMM likelihood features for ', PARAMS['num_seq'],' sequences and ', PARAMS['numMix'],' mixtures.\n')
    
    
