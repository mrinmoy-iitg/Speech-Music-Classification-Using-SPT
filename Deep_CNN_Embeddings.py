#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:14:11 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import datetime
import numpy as np
import configparser
import os
from keras.models import Model
import lib.misc as misc
import librosa
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, Dense, Flatten
from keras import optimizers





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




def get_model(classes, input_shape):
    '''
    Baseline :- papakostas_et_al
    '''
    num_classes = len(classes)
    print('Input shape: ', input_shape)
    
    input_img = Input(shape=input_shape)

    x = Conv2D(96, kernel_size=(5, 5), strides=(2, 2), data_format='channels_first')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(4096)(x)
    x = Activation('relu', name='penultimate_layer')(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)


    model = Model(input_img, output)
    learning_rate = 0.001
    optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics=['accuracy'])    

    print('Baseline of Papakostas et al.\n', model.summary())
    
    return model, learning_rate




def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Deep_CNN_Embeddings.py']
    PARAMS = {
            'folder': section['folder'],
            'modelPath': section['modelPath'],
            'opDir':'',
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'modelName':'',
            'data_generator':section.getboolean('data_generator'),
            'save_flag':section.getboolean('save_flag'),
            'use_GPU':section.getboolean('use_GPU'),
            'GPU_session':None,
            'iterations': int(section['iterations']),
            }
    print(PARAMS)
    
    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()

    PARAMS['classes'], PARAMS['dataset'], PARAMS['preprocessing_type'], PARAMS['feat_type'] = misc.get_classnames(PARAMS['folder'])    
    if os.path.exists(PARAMS['modelPath']):
        for numIter in range(PARAMS['iterations']):
            PARAMS['iter'] = numIter
        
            PARAMS['opDir'] = PARAMS['folder'] + '/__DeepCNN_feat/DCNN_dim=4096_iter' + str(numIter) + '/'    
            if not os.path.exists(PARAMS['opDir']):
                os.makedirs(PARAMS['opDir'])
            
            weightFile = PARAMS['modelPath'] + '/iter0_model.h5'
            architechtureFile = PARAMS['modelPath'] + '/iter0_model.json'
        
            if PARAMS['use_GPU']:
                PARAMS['GPU_session'] = start_GPU_session()
            
            full_model = None
            bottleneck_model = None

            if os.path.exists(weightFile):
                input_shape = (1, 200, 200)
                full_model, learning_rate = get_model(PARAMS['classes'], input_shape)
                full_model.load_weights(weightFile) # Load weights into the new model
                print('CNN model loaded')
        
                '''
                Extracting CNN penultimate layer embeddings
                '''
                layer_name = 'penultimate_layer'
                bottleneck_model = Model(inputs=full_model.input, outputs=full_model.get_layer(layer_name).output)
                
                for i in range(len(PARAMS['classes'])):
                    fold = PARAMS['classes'][i]
            
            	#searching through all the folders and picking up audio
                DIR = misc.get_class_folds(PARAMS['folder'])
                print('DIR: ', DIR)
        
                clNum = -1
                for fold in DIR:
                    clNum += 1
                    deepcnn_embedding_feature_folder = PARAMS['opDir'] + '/' + fold + '/'
                    if not os.path.exists(deepcnn_embedding_feature_folder):
                        os.makedirs(deepcnn_embedding_feature_folder)
                    
                    path = PARAMS['folder'] + '/' + fold + '/'	#path where the original feature vector is
                    print('\n\n\n',path)
                    files = librosa.util.find_files(path, ext=["npy"])
                    numFiles = np.size(files)
                    
                    count = 1	#to see in the terminal how many files have been loaded
                    
                    for f in range(numFiles): #numFiles
                        original_feature_file = files[f]
                        original_feature = np.load(original_feature_file)
                        # original_feature = np.expand_dims(original_feature, axis=0)
                        # print('original_feature_file: ', np.shape(original_feature), original_feature_file)
                        
                        fName = deepcnn_embedding_feature_folder + '/' + original_feature_file.split('/')[-1]
        
                        print('data shape: ', np.shape(original_feature))
                        deepcnn_embedding_FV = bottleneck_model.predict(original_feature)
                        
                        np.save(fName, deepcnn_embedding_FV)
                        print(count,' shape=',np.shape(deepcnn_embedding_FV), ' ', fName)
            else:
                print('Weight file does not exist!!!')
                    
            del full_model
            del bottleneck_model
                    
            if PARAMS['use_GPU']:
                reset_GPU_session()
            
    else:
        print('CNN models do not exist!!!')