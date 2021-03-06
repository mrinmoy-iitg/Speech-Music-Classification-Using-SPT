#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:55:09 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import os
import numpy as np
import librosa
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report, confusion_matrix
import json






def save_obj(obj, folder, name):
    with open(folder+'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    with open(folder+'/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_classnames(folder):
    dirpath, dirnames, filenames = next(os.walk(folder))
    classes = {}
    clCount = 0
    for i in range(len(dirnames)):
        if dirnames[i].startswith('__') or dirnames[i].startswith('.'):
            continue
        classes.setdefault(clCount,[]).append(dirnames[i])
        clCount += 1
    print('Classes: ', classes)

    feat_type = folder.split("/")[-4]
    dataset = folder.split("/")[-3]
    preprocessing_type = folder.split("/")[-2]
    return classes, dataset, preprocessing_type, feat_type



def scale_data(train_data, val_data, test_data):
    if len(np.shape(train_data))==2:
        All_train_data = train_data
        All_train_data = np.append(All_train_data, val_data, 0)
        print('Scale data func: All_train_data=', np.shape(All_train_data))

        std_scale = preprocessing.StandardScaler().fit(All_train_data)

        train_data_scaled = std_scale.transform(train_data)
        val_data_scaled = std_scale.transform(val_data)
        test_data_scaled = std_scale.transform(test_data)

    elif len(np.shape(train_data))>=3:
        All_train_data = train_data
        All_train_data = np.append(All_train_data, val_data, 0)

        All_train_data_mean = All_train_data.mean(axis=(0), keepdims=True)  
        All_train_data_std = All_train_data.std(axis=(0), keepdims=True)
        All_train_data_std[All_train_data_std==0] = 1
            
        train_data_scaled = (train_data-All_train_data_mean)/All_train_data_std
        val_data_scaled = (val_data-All_train_data_mean)/All_train_data_std
        test_data_scaled = (test_data-All_train_data_mean)/All_train_data_std

    return train_data_scaled, val_data_scaled, test_data_scaled



def getPerformance(PtdLabels, GroundTruths):
    ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels).flatten()
    Report = classification_report(y_true=GroundTruths, y_pred=PtdLabels, output_dict=True, labels=np.unique(GroundTruths))
    fscore = []
    Cl = np.unique(GroundTruths)
    Report_keys = [key for key in Report.keys()]
    for i in Cl:
        fscore.append(np.round(Report[Report_keys[int(i)]]['f1-score'], 4))
    mean_fscore = np.round(np.mean(fscore), 4)
    fscore.append(mean_fscore)

    return ConfMat, fscore



def print_results(PARAMS, **kwargs):
    opFile = PARAMS['opDir'] + '/' + PARAMS['clFunc'] + '_performance.csv'

    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = 'iter'
    values = str(PARAMS['iter'])
    for i in range(len(kwargs)):
        heading = heading + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[0]
        values = values + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()




def print_congiguration(PARAMS):
    opFile = PARAMS['opDir'] + '/Configuration.csv'
    fid = open(opFile, 'a+', encoding = 'utf-8')
    PARAM_keys = [key for key in PARAMS.keys()]
    for i in range(len(PARAM_keys)):
        if PARAM_keys[i]=='GPU_session':
            continue
        fid.write(PARAM_keys[i] + '\t')
        fid.write(json.dumps(PARAMS[PARAM_keys[i]]))
        fid.write('\n')
    fid.close()
    
    
    
    
def get_file_list(PARAMS):
    TR_PART = 1-PARAMS['TS_PART']

    file_list_train = {}
    file_list_val = {}
    file_list_test = {}
    totTrainFiles = 0
    totValFiles = 0
    totTestFiles = 0
    for clNum in range(len(PARAMS['classes'])):
        path = PARAMS['folder'] + '/' + PARAMS['classes'][clNum][0]
        files = np.array(os.listdir(path))
        
        idx = np.array(list(range(len(files))))
        np.random.shuffle(idx)
        
        numTrain = int(TR_PART*len(files))
        numVal = int(0.2*numTrain)
        numTrain -= numVal
        
        totTrainFiles += numTrain
        totValFiles += numVal
        totTestFiles += len(files)-numTrain-numVal
        
        file_list_train.setdefault(PARAMS['classes'][clNum][0],[]).append(files[idx[0:numTrain]])
        file_list_val.setdefault(PARAMS['classes'][clNum][0],[]).append(files[idx[numTrain:numTrain+numVal]])
        file_list_test.setdefault(PARAMS['classes'][clNum][0],[]).append(files[idx[numTrain+numVal:]])

        print('totTrainFiles: ', totTrainFiles, 'totTestFiles: ', totTestFiles, 'totValFiles: ', totValFiles)
        
    print('totTrainFiles: ', totTrainFiles, 'totTestFiles: ', totTestFiles, 'totValFiles: ', totValFiles)
    
    FL_Ret = {
            'file_list_train': file_list_train,
            'file_list_val': file_list_val,
            'file_list_test': file_list_test,
            }
    
    return FL_Ret



'''
This function reads in the files listed and returns their contents as a single
2D array
'''
def load_files_cbow_features(data_path, files, seqNum):
    data = np.empty([],dtype=float)
    file_mark = np.empty([],dtype=float)
    data_file_mark = {}
    
    fileCount = 0
    data_marking_start = 0
    data_marking = 0
    for fl in files:
        fName = data_path + '/' + fl
        SPS_matrix = np.load(fName, allow_pickle=True)

        row = SPS_matrix[:, :, seqNum]
        markings = np.ones((1,np.shape(row)[0]))
        markings *= np.shape(row)[1]
        markings = markings.astype(int)
        row = row.ravel()
        data_marking += np.shape(SPS_matrix)[0]
        
        '''
        Storing the row in data array.
        '''
        if np.size(data)<=1:
            data = np.array(row, ndmin=2)
            file_mark = np.array(markings, ndmin=2)
        else:
            data = np.append(data, np.array(row, ndmin=2), 1)
            file_mark = np.append(file_mark, np.array(markings, ndmin=2), 1)
        
        data_file_mark.setdefault(fl,[]).append([data_marking_start, data_marking])
        data_marking_start = data_marking
        
        fileCount += 1
    file_mark = np.cumsum(file_mark).astype(int)
    print('Reading files ... ',fileCount, '/', len(files))
    
    print('Data loaded: ', np.shape(data), np.shape(file_mark))
    return data, file_mark, data_file_mark



def load_data_from_files(PARAMS, files, data_folder):
    label = np.array
    data = np.array
    files_dir = {}
    fileCount = 0
    for fl in files:
        FV = np.load(data_folder + '/' + fl, allow_pickle=True)

        if PARAMS['clFunc']=='CNN': 
            while len(np.shape(FV))<4:
                FV = np.expand_dims(FV, 0)
        
        if fileCount==0:
            data = FV
            file_interval_index_start = 0
            file_interval_index_end = np.shape(data)[0]
        else:
            file_interval_index_start = np.shape(data)[0]
            data = np.append(data, FV, 0)
            file_interval_index_end = np.shape(data)[0]
        
        all_classnames = np.array([val[0] for val in PARAMS['classes'].values()])
        current_class = data_folder.split('/')[-2]
        lab = (all_classnames==current_class).tolist().index(True)
        if fileCount==0:
            label = np.array([lab]*np.shape(FV)[0], ndmin=2)
        else:
            label = np.append(label, np.array([lab]*np.shape(FV)[0], ndmin=2), 1)
        
        files_dir.setdefault(fl,[]).append(list(range(file_interval_index_start, file_interval_index_end)))
        fileCount += 1
    
    label = np.squeeze(label)
    return data, label, files_dir


   
def load_train_val_test_data(PARAMS):
    dirnames = get_class_folds(PARAMS['folder'])
    print('DIR: ', dirnames)

    train_data = np.array
    val_data = np.array
    test_data = np.array
    
    train_label = np.array
    val_label = np.array
    test_label = np.array
    
    train_files = {}
    val_files = {}
    test_files = {}

    for i in range(len(dirnames)):
        dir_i = dirnames[i]
        
        data_folder = PARAMS['folder'] + '/' + dir_i + '/'
        file_list = PARAMS['folder'] + '/file_list_iter' + str(PARAMS['iter']) + '.pkl'
        if os.path.exists(file_list):
            print('File list ' + file_list + ' available !!!')
            FL_Ret = load_obj(PARAMS['folder'], 'file_list_iter' + str(PARAMS['iter']))
        else:
            print('File list ' + file_list + ' NOT available !!!')
            FL_Ret = get_file_list(PARAMS)
            save_obj(FL_Ret, PARAMS['folder'], 'file_list_iter' + str(PARAMS['iter']))
            
        '''
        Added on 19 Dec 2019 for performing generalization performance ~~~~~~~~
        '''
        if PARAMS['generalization_perf']>0:
            file_list_train = FL_Ret['file_list_train'][dir_i][0]
            idx = list(range(len(file_list_train)))
            np.random.shuffle(idx)
            numTrainFiles = int(PARAMS['generalization_perf']*len(idx))
            FL_Ret['file_list_train'][dir_i][0] = file_list_train[idx[:numTrainFiles]]
        '''
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        
        print('Loading training data ')
        train_data_dir, train_label_dir, train_files_dir = load_data_from_files(PARAMS, FL_Ret['file_list_train'][dir_i][0], data_folder)
            
        print('Loading validation data ')
        val_data_dir, val_label_dir, val_files_dir = load_data_from_files(PARAMS, FL_Ret['file_list_val'][dir_i][0], data_folder)

        print('Loading testing data ')
        test_data_dir, test_label_dir, test_files_dir = load_data_from_files(PARAMS, FL_Ret['file_list_test'][dir_i][0], data_folder)
            
        train_files.setdefault(dir_i, []).append(train_files_dir)
        val_files.setdefault(dir_i, []).append(val_files_dir)
        test_files.setdefault(dir_i, []).append(test_files_dir)

        if len(dirnames)==1:
            train_data = train_data_dir
            val_data = val_data_dir
            test_data = test_data_dir
            
            train_label = np.array(train_label_dir)
            val_label = np.array(val_label_dir)
            test_label = np.array(test_label_dir)
        else:
            if i==0:
                train_data = train_data_dir
                val_data = val_data_dir
                test_data = test_data_dir

                train_label = np.ones(len(train_label_dir))*i
                val_label = np.ones(len(val_label_dir))*i
                test_label = np.ones(len(test_label_dir))*i
            else:
                train_data = np.append(train_data, train_data_dir, 0)
                val_data = np.append(val_data, val_data_dir, 0)
                test_data = np.append(test_data, test_data_dir, 0)

                train_label = np.append(train_label, np.ones(len(train_label_dir))*i)
                val_label = np.append(val_label, np.ones(len(val_label_dir))*i)
                test_label = np.append(test_label, np.ones(len(test_label_dir))*i)

    if PARAMS['save_flag']:
        np.save(PARAMS['folder']+'train_data_iter'+str(PARAMS['iter'])+'.npy', train_data)
        np.save(PARAMS['folder']+'train_label_iter'+str(PARAMS['iter'])+'.npy', train_label)

        np.save(PARAMS['folder']+'val_data_iter'+str(PARAMS['iter'])+'.npy', val_data)
        np.save(PARAMS['folder']+'val_label_iter'+str(PARAMS['iter'])+'.npy', val_label)

        np.save(PARAMS['folder']+'test_data_iter'+str(PARAMS['iter'])+'.npy', test_data)
        np.save(PARAMS['folder']+'test_label_iter'+str(PARAMS['iter'])+'.npy', test_label)

        save_obj(train_files, PARAMS['folder'], 'train_files_iter'+str(PARAMS['iter']))
        save_obj(val_files, PARAMS['folder'], 'val_files_iter'+str(PARAMS['iter']))
        save_obj(test_files, PARAMS['folder'], 'test_files_iter'+str(PARAMS['iter']))
    
    print('Train data shape: ', train_data.shape, train_label.shape)
    print('Val data shape: ', val_data.shape, val_label.shape)
    print('Test data shape: ', test_data.shape, test_label.shape)

    return train_data, train_label, val_data, val_label, test_data, test_label, train_files, val_files, test_files





'''
Read all data from folder and split it into train, validation and test data
'''
def get_data(PARAMS):
    savedData = PARAMS['folder']+'/train_data_iter'+str(PARAMS['iter'])+'.npy'
    print('Saved data: ', savedData)
    if not os.path.exists(savedData):
        print('Training data not available !!!')
        train_data, train_label, val_data, val_label, test_data, test_label, train_files, val_files, test_files = load_train_val_test_data(PARAMS)
        
    else:
        train_data = np.array(np.load(PARAMS['folder']+'/train_data_iter'+str(PARAMS['iter'])+'.npy', allow_pickle=True))
        train_label = np.array(np.load(PARAMS['folder']+'/train_label_iter'+str(PARAMS['iter'])+'.npy', allow_pickle=True))

        val_data = np.array(np.load(PARAMS['folder']+'/val_data_iter'+str(PARAMS['iter'])+'.npy', allow_pickle=True))
        val_label = np.array(np.load(PARAMS['folder']+'/val_label_iter'+str(PARAMS['iter'])+'.npy', allow_pickle=True))

        test_data = np.array(np.load(PARAMS['folder']+'/test_data_iter'+str(PARAMS['iter'])+'.npy', allow_pickle=True))
        test_label = np.array(np.load(PARAMS['folder']+'/test_label_iter'+str(PARAMS['iter'])+'.npy', allow_pickle=True))

        train_files_fName = 'train_files_iter'+str(PARAMS['iter'])
        val_files_fName = 'val_files_iter'+str(PARAMS['iter'])
        test_files_fName = 'test_files_iter'+str(PARAMS['iter'])
            
        train_files = load_obj(PARAMS['folder'], train_files_fName)
        val_files = load_obj(PARAMS['folder'], val_files_fName)
        test_files = load_obj(PARAMS['folder'], test_files_fName)
            
        
    if PARAMS['DIM']=='':
        PARAMS['DIM'] = list(range(np.shape(train_data)[-1]))

    if not PARAMS['clFunc'] == 'CNN':
        train_data = np.squeeze(train_data)
    if len(np.shape(train_data))<3:
        print('Shape: ', np.shape(train_data), len(PARAMS['DIM']))
        train_data = train_data[:, PARAMS['DIM']]

    if not PARAMS['clFunc'] == 'CNN':
        val_data = np.squeeze(val_data)
    if len(np.shape(val_data))<3:
        val_data = val_data[:, PARAMS['DIM']]

    if not PARAMS['clFunc'] == 'CNN':
        test_data = np.squeeze(test_data)
    if len(np.shape(test_data))<3:
        test_data = test_data[:, PARAMS['DIM']]

    print(np.shape(train_data), np.shape(val_data), np.shape(test_data))
   
    # DATA BALANCING ----------------------------------------------------------
    if PARAMS['data_balancing']:
        print('Unbalanced data: ', np.shape(train_data))
        # Over and under sampling
        smote_enn = SMOTEENN(sampling_strategy=1.0)
        train_data, train_label = smote_enn.fit_resample(train_data, train_label)
        print('Balanced data: ', np.shape(train_data))
    # -------------------------------------------------------------------------
    

    '''
    Scaling the data
    '''
    if PARAMS['scale_data']:
        train_data, val_data, test_data = scale_data(train_data, val_data, test_data)


    print('Data shape: ', np.shape(train_data), np.shape(val_data), np.shape(test_data))
    data_dict = {
            'train_data': train_data,
            'train_label': train_label,
            'train_files': train_files,
            'val_data': val_data,
            'val_label': val_label,
            'val_files': val_files,
            'test_data': test_data,
            'test_label': test_label,
            'test_files': test_files,
            }
    return data_dict




def pca_dim_reduction(train_data, val_data, test_data):
    print('Before PCA: ', np.shape(train_data), np.shape(val_data), np.shape(test_data))
    from sklearn.decomposition import PCA
    dim_reduction = PCA(n_components=0.85, whiten=False)
    All_train_data = np.append(train_data, val_data, 0)
    
    dim_reduction.fit(All_train_data)
    
    train_data = dim_reduction.transform(train_data)
    val_data = dim_reduction.transform(val_data)
    test_data = dim_reduction.transform(test_data)
    print('After PCA: ', np.shape(train_data), np.shape(val_data), np.shape(test_data))
    
    return train_data, val_data, test_data
    




def get_class_folds(path):
    base_path, DIR, filenames = next(os.walk(path))
    DIR_temp = []
    for dir_i in DIR:
        if not dir_i.startswith('__') and not dir_i.startswith('.'):
            DIR_temp.append(dir_i)
    DIR = DIR_temp
    
    return DIR
    
