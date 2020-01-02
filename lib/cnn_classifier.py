#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:00:47 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
import os
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input
import lib.misc as misc
import lib.deep_learning as dplearn
from keras.initializers import RandomNormal, Constant
from keras.models import Model



def get_model(classes, input_shape):
    model = Sequential()

    '''
    Baseline :- papakostas_et_al
    '''
    num_classes = len(classes)
    print('Input shape: ', input_shape)
    
    input_img = Input(shape=input_shape)
    
    x = Conv2D(96, kernel_size=(5, 5), strides=(2, 2), data_format='channels_first', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), data_format='channels_first', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), data_format='channels_first', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    
    x = Dense(4096, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(4096, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)


    model = Model(input_img, output)
    learning_rate = 0.001
    optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics=['accuracy'])    

    print('Baseline of Papakostas et al.\n', model.summary())
    
    return model, learning_rate



def train_cnn(PARAMS, data_dict):
    weightFile = PARAMS['modelName'].split('.')[0] + '.h5'
    architechtureFile = PARAMS['modelName'].split('.')[0] + '.json'
    paramFile = PARAMS['modelName'].split('.')[0] + '_params.npz'
    
    FL_Ret = {}
    
    if not os.path.exists(weightFile):
        epochs = 16
        batch_size = 128
        input_shape = (1, 200, 200)
        model, learning_rate = get_model(PARAMS['classes'], input_shape)

        model, trainingTimeTaken, FL_Ret, History = dplearn.train_model(PARAMS, data_dict, model, epochs, batch_size, weightFile)
        
        if PARAMS['save_flag']:
            model.save_weights(weightFile) # Save the weights
            with open(architechtureFile, 'w') as f: # Save the model architecture
                f.write(model.to_json())
            np.savez(paramFile, epochs=epochs, batch_size=batch_size, input_shape=input_shape, lr=learning_rate, trainingTimeTaken=trainingTimeTaken)
            misc.save_obj(History, PARAMS['opDir'], 'training_history_iter'+str(PARAMS['iter']))
            misc.save_obj(FL_Ret, PARAMS['opDir'], 'FL_Ret')
        print('CNN model trained.')
    else:
        epochs = np.load(paramFile)['epochs']
        batch_size = np.load(paramFile)['batch_size']
        input_shape = np.load(paramFile)['input_shape']
        learning_rate = np.load(paramFile)['lr']
        trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
        optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9)
        
        with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
            model = model_from_json(f.read())
        model.load_weights(weightFile) # Load weights into the new model
        History = misc.load_obj(PARAMS['opDir'], 'training_history_iter'+str(PARAMS['iter']))
        FL_Ret = misc.load_obj(PARAMS['opDir'], 'FL_Ret')

        model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics=['accuracy'])

        print('CNN model exists! Loaded. Training time required=',trainingTimeTaken)
      
        
    Train_Params = {
            'model': model,
            'History': History,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'FL_Ret': FL_Ret,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params
        


def test_cnn(PARAMS, data_dict, Train_Params):
    loss, performance, testingTimeTaken, ConfMat, fscore, PtdLabels_test_interval, Predictions_test_interval = dplearn.test_model(PARAMS, data_dict, Train_Params)

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
