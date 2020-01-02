#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:43:50 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import time
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import numpy as np
import lib.misc as misc



def generator(file_list, PARAMS, batchSize):
    numFiles = np.max([int(batchSize/len(PARAMS['classes'])),1])
    batch_count = 0
    while 1:
        batchData = np.empty([], dtype=float)
        batchLabel = np.empty([], dtype=float)

        for clNum in range(len(PARAMS['classes'])):
            fold = PARAMS['classes'][clNum][0]
            files = np.squeeze(file_list[fold])
            
            randIdx = list(range(len(files)))
            np.random.shuffle(randIdx)
            batch_files = files[randIdx[:numFiles+1]]
            
            labCount = 0
            for batchNum in range(numFiles):
                fName = PARAMS['folder'] + fold + '/' + batch_files[batchNum]
                fv = np.array(np.load(fName, allow_pickle=True), ndmin=2)
                if PARAMS['clFunc']=='CNN':
                    while len(np.shape(fv))<4:
                        fv = np.expand_dims(fv, 1)
                if np.size(batchData)<=1:
                    batchData = fv
                else:
                    batchData = np.append(batchData, fv, axis=0)
                labCount += np.shape(fv)[0]
            
            if clNum==0:
                batchLabel = np.ones(shape=(1, labCount))*clNum
            else:
                lab = np.ones(shape=(1, labCount))*clNum
                batchLabel = np.append(batchLabel, lab)
        
        OHE_batchLabel = to_categorical(batchLabel)
        OHE_batchLabel = np.squeeze(OHE_batchLabel)
        
#        print('batch data shape: ', np.shape(batchData), np.shape(OHE_batchLabel), np.unique(batchLabel))
        if not np.shape(batchData)[0]==np.shape(OHE_batchLabel)[0]:
            continue
        
        batch_count += 1
        yield batchData, OHE_batchLabel



def generator_test(file_name, PARAMS, clNum):
    fold = PARAMS['classes'][clNum][0]
    fName = PARAMS['folder'] + '/' + fold + '/' + file_name
    batchData = np.array(np.load(fName), ndmin=2)
    if PARAMS['clFunc']=='CNN':
        while len(np.shape(batchData))<4:
            batchData = np.expand_dims(batchData, 1)
    numLab = np.shape(batchData)[0]

    batchLabel = np.ones(shape=(numLab))*clNum
#    OHE_batchLabel = to_categorical(batchLabel)

    return batchData, batchLabel



def train_model(PARAMS, data_dict, model, epochs, batch_size, weightFile): # Updated 23-05-2019
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    logFile = '/'.join(weightFile.split('/')[:-2]) + '/log_iter' + str(PARAMS['iter']) + '.csv'
    csv_logger = CSVLogger(logFile)

    FL_Ret = {}
    trainingTimeTaken = 0
    start = time.clock()
    if not PARAMS['data_generator']:
        train_data = data_dict['train_data']
        val_data = data_dict['val_data']
        OHE_trainLabel = to_categorical(data_dict['train_label'])
        OHE_valLabel = to_categorical(data_dict['val_label'])
        
        # Train the model
        History = model.fit(
                x=train_data,
                y=OHE_trainLabel, 
                epochs=epochs,
                batch_size=batch_size, 
                verbose=1,
                validation_data = (val_data, OHE_valLabel),
                callbacks=[csv_logger, es, mcp],
                shuffle=True,
                )
    else:
        FL_Ret = misc.get_file_list(PARAMS)

        SPE = PARAMS['train_steps_per_epoch']
        SPE_val = PARAMS['val_steps']
        print('SPE: ', SPE,SPE_val)
        
        # Train the model
        History = model.fit_generator(
                generator(FL_Ret['file_list_train'], PARAMS, batch_size),
                steps_per_epoch = SPE,
                validation_data = generator(FL_Ret['file_list_val'], PARAMS, batch_size), 
                validation_steps = SPE_val,
                epochs=epochs, 
                verbose=1,
                callbacks=[csv_logger, es, mcp],
                shuffle=True,
                )

    trainingTimeTaken = time.clock() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, FL_Ret, History



def test_model(PARAMS, data_dict, Train_Params):
    loss = 0
    performance = 0
    testingTimeTaken = 0
    PtdLabels = []
    test_data = data_dict['test_data']
    
    start = time.clock()
    if not PARAMS['data_generator']:
        OHE_testLabel = to_categorical(data_dict['test_label'])
        loss, performance = Train_Params['model'].evaluate(x=test_data, y=OHE_testLabel)
        Predictions = Train_Params['model'].predict(test_data)
        PtdLabels = np.argmax(Predictions, axis=1)
        GroundTruth = data_dict['test_label']
        
    else:
        class_wise_numFiles = [len(files[0]) for files in Train_Params['FL_Ret']['file_list_test'].values()]
        totTestFiles = np.sum(class_wise_numFiles)
        SPE = int(totTestFiles/Train_Params['batch_size'])
        loss, performance = Train_Params['model'].evaluate_generator(
                generator(Train_Params['FL_Ret']['file_list_test'], PARAMS, Train_Params['batch_size']),
                steps=SPE, 
                verbose=1
                )
        
        PtdLabels = []
        GroundTruth = []
        count = -1
        Predictions = np.empty([])
        file_keys = [key for key in Train_Params['FL_Ret']['file_list_test'].keys()]
        for clNum in range(len(file_keys)):
            files = Train_Params['FL_Ret']['file_list_test'][file_keys[clNum]][0]
            for fl in files:
                count += 1
                file_name = fl
                batchData, batchLabel = generator_test(file_name, PARAMS, clNum)
                pred = Train_Params['model'].predict(x=batchData)
                pred_lab = np.argmax(pred, axis=1)
                PtdLabels.extend(pred_lab)
                GroundTruth.extend(batchLabel)
                if np.size(Predictions)<=1:
                    Predictions = pred
                else:
                    Predictions = np.append(Predictions, pred, 0)

    testingTimeTaken = time.clock() - start
    print('Time taken for model testing: ',testingTimeTaken)
    ConfMat, fscore = misc.getPerformance(PtdLabels, GroundTruth)
    
    return loss, performance, testingTimeTaken, ConfMat, fscore, PtdLabels, Predictions    

