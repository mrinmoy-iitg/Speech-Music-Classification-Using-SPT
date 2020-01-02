#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:30:21 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import datetime
import numpy as np
import configparser
import os
from keras.layers import Dense, BatchNormalization, Activation, Input
from keras.optimizers import Adam
from keras.models import Model
import lib.misc as misc
import lib.deep_learning as dplearn
from keras.models import model_from_json
import librosa
from sklearn import preprocessing




def start_GPU_session():
    import tensorflow as tf
    import keras
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    return sess



def reset_GPU_session():
    import keras
    keras.backend.clear_session()



def get_BN_network_model(input_shape, bn_shape, output_shape):
    input_layer = Input(shape=input_shape)
    input_dim = input_shape[0]

    hidden_1 = Dense(input_dim, input_dim=input_dim)(input_layer)
    hidden_1 = BatchNormalization(axis=1, scale=True)(hidden_1)
    hidden_1 = Activation('sigmoid', name='hidden_1')(hidden_1)

    hidden_2 = Dense(input_dim)(hidden_1)
    hidden_2 = BatchNormalization(axis=1, scale=True)(hidden_2)
    hidden_2 = Activation('sigmoid', name='hidden_2')(hidden_2)

    bottleneck_1 = Dense(bn_shape)(hidden_2)
    bottleneck_1 = BatchNormalization(axis=1, scale=True)(bottleneck_1)
    bottleneck_1 = Activation('sigmoid', name='bottleneck_1')(bottleneck_1)

    hidden_4 = Dense(input_dim)(bottleneck_1)
    hidden_4 = BatchNormalization(axis=1, scale=True)(hidden_4)
    hidden_4 = Activation('sigmoid', name='hidden_4')(hidden_4)

    hidden_5 = Dense(input_dim)(hidden_4)
    hidden_5 = BatchNormalization(axis=1, scale=True)(hidden_5)
    hidden_5 = Activation('sigmoid', name='hidden_5')(hidden_5)

    output_layer = Dense(output_shape)(hidden_5)
    output_layer = Activation('softmax')(output_layer)
    
    model = Model(input_layer, output_layer)
    
    learning_rate = 0.005
    adam = Adam(lr=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    return model, learning_rate
    




def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['compute_bottleneck_features.py']
    PARAMS = {
            'folder': section['folder'],
            'opDir':'',
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'BN_size': int(section['BN_size']),
            'dataset':section['dataset'],
            'modelName':'',
            'data_generator':section.getboolean('data_generator'),
            'save_flag':section.getboolean('save_flag'),
            'TS_PART':0.2,
            'clFunc':'DNN',
            'use_GPU':section.getboolean('use_GPU'),
            'train_steps_per_epoch':int(section['train_steps_per_epoch']),
            'val_steps':int(section['val_steps']),
            'GPU_session':None,
            'iterations': int(section['iterations']),
            'dim_range1': int(section['dim_range1']),
            'dim_range2': int(section['dim_range2']),
            'DIM':'',
            'data_balancing': False,
            'scale_data': section.getboolean('scale_data'),
            'PCA_flag': False,
            }
    print(PARAMS)
    PARAMS['DIM'] = list(range(PARAMS['dim_range1'], PARAMS['dim_range2']))
    print('DIM: ',PARAMS['DIM'])
    
    return PARAMS





if __name__ == '__main__':
    PARAMS = __init__()

    PARAMS['classes'], PARAMS['dataset'], PARAMS['preprocessing_type'], PARAMS['feat_type'] = misc.get_classnames(PARAMS['folder'])    
    modelPath = PARAMS['folder'] + '/__BN_models/'
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    for numIter in range(PARAMS['iterations']):
        PARAMS['iter'] = numIter
    
        PARAMS['opDir'] = PARAMS['folder'] + '/__BN_feat_dim' + str(PARAMS['BN_size']) + '_iter' + str(numIter) + '/'    
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
            
        PARAMS['modelName'] = modelPath + '/BN_model_iter' + str(numIter) + '.xyz'
        
        weightFile = PARAMS['modelName'].split('.')[0] + '.h5'
        architechtureFile = PARAMS['modelName'].split('.')[0] + '.json'
        paramFile = PARAMS['modelName'].split('.')[0] + '_params.npz'
    
        FL_Ret = {}
        
        if PARAMS['use_GPU']:
            PARAMS['GPU_session'] = start_GPU_session()
    


        '''
        Load data
        '''
        if not PARAMS['data_generator']:
            data_dict = misc.get_data(PARAMS)
        else:
            data_dict = {'train_data':np.empty, 'train_label':np.empty, 'val_data':np.empty, 'val_label':np.empty, 'test_data':np.empty, 'test_label':np.empty}
    
    
        '''
        Data scaling is done locally so that later while generating bottleneck 
        features, original features can be scaled and then passed to the bottleneck network
        '''
        std_scale = None
        if not PARAMS['scale_data']:
            All_train_data = data_dict['train_data']
            All_train_data = np.append(All_train_data, data_dict['val_data'], 0)
    
            std_scale = preprocessing.StandardScaler().fit(All_train_data)
    
            train_data_scaled = std_scale.transform(data_dict['train_data'])
            val_data_scaled = std_scale.transform(data_dict['val_data'])
            test_data_scaled = std_scale.transform(data_dict['test_data'])
            
            data_dict['train_data'] = train_data_scaled
            data_dict['val_data'] = val_data_scaled
            data_dict['test_data'] = test_data_scaled
        '''
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''

        
        if not os.path.exists(weightFile):
            epochs = 200
            batch_size = 10
            input_shape = (len(PARAMS['DIM']),)
            bn_shape = PARAMS['BN_size']
            output_shape = 2
            full_model, learning_rate = get_BN_network_model(input_shape, bn_shape, output_shape)
    
            full_model, trainingTimeTaken, FL_Ret, History = dplearn.train_model(PARAMS, data_dict, full_model, epochs, batch_size, weightFile)
            
            if PARAMS['save_flag']:
                full_model.save_weights(weightFile) # Save the weights
                with open(architechtureFile, 'w') as f: # Save the model architecture
                    f.write(full_model.to_json())
                np.savez(paramFile, epochs=epochs, batch_size=batch_size, input_shape=input_shape, lr=learning_rate, trainingTimeTaken=trainingTimeTaken)
                # misc.save_obj(History, PARAMS['opDir'], 'training_history')
                # misc.save_obj(FL_Ret, PARAMS['opDir'], 'FL_Ret')
            print('Bottleneck feature extraction model trained.')
            
        else:
            epochs = np.load(paramFile)['epochs']
            batch_size = np.load(paramFile)['batch_size']
            input_shape = np.load(paramFile)['input_shape']
            learning_rate = np.load(paramFile)['lr']
            trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
            optimizer = Adam(lr=learning_rate)
            
            with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
                full_model = model_from_json(f.read())
            full_model.load_weights(weightFile) # Load weights into the new model
            # History = misc.load_obj(PARAMS['opDir'], 'training_history')
            # FL_Ret = misc.load_obj(PARAMS['opDir'], 'FL_Ret')
    
            full_model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    
            print('Bottleneck feature extraction model exists! Loaded. Training time taken=',trainingTimeTaken)
    
        
        '''
        Computing Bottleneck features for all data
        '''
        layer_name = 'bottleneck_1'
        bottleneck_model = Model(inputs=full_model.input, outputs=full_model.get_layer(layer_name).output)
        
        for i in range(len(PARAMS['classes'])):
            fold = PARAMS['classes'][i]
    
        
    	#searching through all the folders and picking up audio
        DIR = misc.get_class_folds(PARAMS['folder'])
        print('DIR: ', DIR)

        clNum = -1
        for fold in DIR:
            clNum += 1
            bn_feature_folder = PARAMS['opDir'] + '/' + fold + '/'
            if not os.path.exists(bn_feature_folder):
                os.makedirs(bn_feature_folder)
            
            path = PARAMS['folder'] + '/' + fold + '/'	#path where the original feature vector is
            print('\n\n\n',path)
            files = librosa.util.find_files(path, ext=["npy"])
            numFiles = np.size(files)
            
            count = 1	#to see in the terminal how many files have been loaded
            
            for f in range(numFiles): #numFiles
                original_feature_file = files[f]
                original_feature = np.load(original_feature_file)
                if not PARAMS['scale_data']:
                    original_feature = std_scale.transform(original_feature)
                
                fName = bn_feature_folder + '/' + original_feature_file.split('/')[-1]

                print('data shape: ', np.shape(original_feature))
                bottleneck_FV = bottleneck_model.predict(original_feature)
                
                np.save(fName, bottleneck_FV)
                print(count, ' shape=', np.shape(bottleneck_FV), np.shape(original_feature), ' ', fName)
                
        if PARAMS['use_GPU']:
            reset_GPU_session()
        
        full_model = None
        bottleneck_model = None
        data_dict = None
        del full_model
        del bottleneck_model
        del data_dict
