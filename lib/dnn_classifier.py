#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:43:57 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation, Input, Dropout
from keras.optimizers import Adam
from keras.models import model_from_json
import lib.misc as misc
import lib.deep_learning as dplearn
from keras.initializers import RandomNormal, Constant
from keras.models import Model




# define base model
def dnn_model(input_dim, output_dim):
    # create model
    
    input_layer = Input((input_dim,))
    
    layer1_size = input_dim
    dense_1 = Dense(layer1_size, input_dim=(input_dim,), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(input_layer)
    batchnorm_1 = BatchNormalization()(dense_1)
    dropout_1 = Dropout(0.4)(batchnorm_1)
    activation_1 = Activation('relu')(dropout_1)
    
    layer2_size = layer1_size*2
    dense_2 = Dense(layer2_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_1)
    batchnorm_2 = BatchNormalization()(dense_2)
    dropout_2 = Dropout(0.4)(batchnorm_2)
    activation_2 = Activation('relu')(dropout_2)

    layer3_size = int(layer2_size*2/3)
    dense_3 = Dense(layer3_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_2)
    batchnorm_3 = BatchNormalization()(dense_3)
    dropout_3 = Dropout(0.4)(batchnorm_3)
    activation_3 = Activation('relu')(dropout_3)

    layer4_size = int(layer3_size/2)
    dense_4 = Dense(layer4_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_3)
    batchnorm_4 = BatchNormalization()(dense_4)
    dropout_4 = Dropout(0.4)(batchnorm_4)
    activation_4 = Activation('relu')(dropout_4)

    layer5_size = int(layer4_size/3)
    dense_5 = Dense(layer5_size, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_4)
    batchnorm_5 = BatchNormalization()(dense_5)
    dropout_5 = Dropout(0.4)(batchnorm_5)
    activation_5 = Activation('relu')(dropout_5)
    
    output_layer = Dense(output_dim, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(activation_5)
    output_layer = Activation('softmax')(output_layer)

    model = Model(input_layer, output_layer)
    
    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)

    optimizerName = 'Adam'
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model, optimizerName, learning_rate





'''
This function is the driver function for learning and evaluating a DNN model. 
The arguments are passed as a dictionary object initialized with the required
data. It returns a dictinary that contains the trained model and other required
information.
'''
def train_dnn(PARAMS, data_dict): # Updated on 25-05-2019
    
    # set remaining variables
    epochs = 100
    batch_size = 64

    weightFile = PARAMS['modelName'].split('.')[0] + '.h5'
    architechtureFile = PARAMS['modelName'].split('.')[0] + '.json'
    paramFile = PARAMS['modelName'].split('.')[0] + '_params.npz'
    
    if not PARAMS['data_generator']:
        PARAMS['input_dim'] = np.shape(data_dict['train_data'])[1]
    else:
        PARAMS['input_dim'] = len(PARAMS['DIM'])

    output_dim = len(PARAMS['classes'])
    print(output_dim)
    FL_Ret = {}

    print('Weight file: ', weightFile, PARAMS['input_dim'], output_dim)
    if not os.path.exists(weightFile):
        model, optimizerName, learning_rate = dnn_model(PARAMS['input_dim'], output_dim)
        print(model.summary())
        
        model, trainingTimeTaken, FL_Ret, History = dplearn.train_model(PARAMS, data_dict, model, epochs, batch_size, weightFile)
        if PARAMS['save_flag']:
            # Save the weights
            model.save_weights(weightFile)
            # Save the model architecture
            with open(architechtureFile, 'w') as f:
                f.write(model.to_json())
            np.savez(paramFile, ep=str(epochs), bs=str(batch_size), lr=str(learning_rate), TTT=str(trainingTimeTaken))
            misc.save_obj(History, PARAMS['opDir'], 'training_history')
    else:
        if os.path.exists(paramFile):
            epochs = int(np.load(paramFile)['ep'])
            batch_size = int(np.load(paramFile)['bs'])
            learning_rate = float(np.load(paramFile)['lr'])
            trainingTimeTaken = float(np.load(paramFile)['TTT'])
            optimizerName = 'Adam'

            # Model reconstruction from JSON file
            with open(architechtureFile, 'r') as f:
                model = model_from_json(f.read())
            # Load weights into the new model
            model.load_weights(weightFile)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=['accuracy'])
            History = misc.load_obj(PARAMS['opDir'], 'training_history')
    
            print('DNN model exists! Loaded. Training time required=',trainingTimeTaken)
            print(model.summary())
    
    Train_Params = {
            'model': model,
            'History': History,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizerName': optimizerName,
            'FL_Ret': FL_Ret,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params



def test_dnn(PARAMS, data_dict, Train_Params):
    loss, performance, testingTimeTaken, ConfMat, fscore, PtdLabels_test_interval, Predictions_test_interval = dplearn.test_model(PARAMS, data_dict, Train_Params)
    
    print('loss: ', loss)
    print('performancermance: ', performance)

    Test_Params = {
        'loss': loss,
        'performance': performance,
        'testingTimeTaken': testingTimeTaken,
        'ConfMat': ConfMat,
        'fscore': fscore,
        'PtdLabels_test': PtdLabels_test_interval,
        'Predictions_test': Predictions_test_interval,
        }

    return Test_Params
