#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:51:23 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
import configparser




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



def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Perform_Classification.py']
    PARAMS = {
            'folder': section['folder'],
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'DIM': [],
            'command': 'Classify',
            'featName': section['featName'],
            'clFunc': section['clFunc'], #GMM, SVM, DNN, CNN
            'TS_PART': float(section['TS_PART']),
            'dim_range1': int(section['dim_range1']),
            'dim_range2': int(section['dim_range2']),
            'iterations': int(section['iterations']),
            'save_flag': section.getboolean('save_flag'),
            'data_generator': section.getboolean('data_generator'),
            'data_balancing': section.getboolean('data_balancing'),
            'use_GPU': section.getboolean('use_GPU'),
            'train_steps_per_epoch':int(section['train_steps_per_epoch']),
            'val_steps':int(section['val_steps']),
            'scale_data': section.getboolean('scale_data'),
            'PCA_flag': section.getboolean('PCA_flag'),
            'GPU_session':None,
            'output_folder':'',
            'preprocessing_type':'',
            'classes':{},
            'dataset':'',
            'feat_type':'',
            'opDir':'',
            'iter':0,
            'modelName':'',
            'input_dim':0,
            'generalization_perf': float(section['generalization_perf'])
            }

    PARAMS['DIM'] = list(range(PARAMS['dim_range1'], PARAMS['dim_range2']))
    print('DIM: ',PARAMS['DIM'])

    return PARAMS





if __name__ == '__main__':
    PARAMS = __init__()

    if PARAMS['featName'].startswith('CBoW'):
        base_folder = PARAMS['folder'] # Temporary

    for numIter in range(PARAMS['iterations']):
        PARAMS['iter'] = numIter

        '''
        Only for CBoW features
        '''
        if PARAMS['featName'].startswith('CBoW'):
            PARAMS['folder'] = base_folder + str(numIter) + '/'
            print('folder: ', PARAMS['folder'])
        '''
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''

        '''
        Initializations
        '''
        PARAMS['classes'], PARAMS['dataset'], PARAMS['preprocessing_type'], PARAMS['feat_type'] = misc.get_classnames(PARAMS['folder'])
        print('Classnames: ', [clName[0] for clName in PARAMS['classes'].values()], ' Dataset: ', PARAMS['dataset'], ' Feature type:', PARAMS['feat_type'])

        if PARAMS['featName'].startswith('CBoW'):
            PARAMS['output_folder'] = '/'.join(PARAMS['folder'].split('/')[:-2]) + '/__RESULTS/' + PARAMS['today'] + '/'
        else:
            PARAMS['output_folder'] = PARAMS['folder'] + '/__RESULTS/' + PARAMS['today'] + '/'
        if not os.path.exists(PARAMS['output_folder']):
            os.makedirs(PARAMS['output_folder'])

        PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['featName'] + '/' + PARAMS['clFunc'] + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
        misc.print_congiguration(PARAMS)


        if PARAMS['use_GPU']:
            PARAMS['GPU_session'] = start_GPU_session()

        '''
        Load data
        '''
        if not PARAMS['data_generator']:
            data_dict = misc.get_data(PARAMS)

            if PARAMS['PCA_flag']:
                data_dict['train_data'], data_dict['val_data'], data_dict['test_data'] = misc.pca_dim_reduction(data_dict['train_data'], data_dict['val_data'], data_dict['test_data'])
        else:
            data_dict = {'train_data':np.empty, 'train_label':np.empty, 'val_data':np.empty, 'val_label':np.empty, 'test_data':np.empty, 'test_label':np.empty}

    
        '''
        Set training parameters
        '''
        if PARAMS['data_generator']:
            input_dim = PARAMS['dim_range2']
        else:
            input_dim = np.shape(data_dict['train_data'])[1]
        print('Input dim: ', input_dim)

        PARAMS['modelName'] = PARAMS['opDir'] + '/iter' + str(PARAMS['iter']) + '_model.xyz'


        # ---------------------------------------------------------------------
        ### -------------------------------- SVM ------------------------------
        # ---------------------------------------------------------------------
        if PARAMS['clFunc']=='SVM':
            import lib.svm_classifier as SVM
            
            PARAMS['modelName'] = PARAMS['opDir'] + '/SVM_' + PARAMS['featName'] + '_iter' + str(PARAMS['iter']) + '.pkl'
            Train_Params, Test_Params = SVM.grid_search_svm(PARAMS, data_dict)
            kwargs = {
                    '0':'optC:'+Train_Params['optC'],
                    '1':'optGamma:'+Train_Params['optGamma'],
                    '2':'Accuracy:'+str(Test_Params['Performance_test'][0]),
                    '3':'F_score_mu:'+str(Test_Params['Performance_test'][1]),
                    '4':'F_score_sp:'+str(Test_Params['Performance_test'][2]),
                    '5':'F_score_avg:'+str(Test_Params['Performance_test'][3]),
                    '6':'featName:'+str(PARAMS['featName']),
                    }
            for i in range(len(PARAMS['classes'])):
                kwargs.setdefault(str(len(kwargs)),[]).append(PARAMS['classes'][i][0]+'SV:' + str(Train_Params['countSV'][i]))
            for i in range(len(PARAMS['classes'])):
                kwargs.setdefault(str(len(kwargs)),[]).append(PARAMS['classes'][i][0]+'TrPts:' + str(Train_Params['countTrPts'][i]))
            misc.print_results(PARAMS, **kwargs)
        # ---------------------------------------------------------------------



        # ---------------------------------------------------------------------
        ### -------------------------------- GMM ------------------------------
        # ---------------------------------------------------------------------
        elif PARAMS['clFunc']=='GMM':
            import lib.gmm_classifier as GMM

            PARAMS['modelName'] = PARAMS['opDir'] + '/GMM_' + PARAMS['featName'] + '_iter' + str(PARAMS['iter']) + '.pkl'
            # Performing Grid-Search --------------------------------------
            score_test, PtdLabels_test, K, Performance_train, Performance_test = GMM.grid_search_gmm(PARAMS, data_dict)
            kwargs = {
                    '0':'K:'+str(K),
                    '1':'Accuracy:'+str(Performance_test[0]),
                    '2':'F_score_mu:'+str(Performance_test[1]),
                    '3':'F_score_sp:'+str(Performance_test[2]),
                    '4':'F_score_avg:'+str(Performance_test[3]),
                    '5':'featName:'+str(PARAMS['featName']),
                    }
            misc.print_results(PARAMS, **kwargs)
                # -------------------------------------------------------------
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        ### -------------------------------- Naive Bayes ----------------------
        # ---------------------------------------------------------------------
        elif PARAMS['clFunc']=='NB':
            import lib.NB_classifier as NB

            PARAMS['modelName'] = PARAMS['opDir'] + '/NB_' + PARAMS['featName'] + '_iter' + str(PARAMS['iter']) + '.pkl'
            # Performing Grid-Search --------------------------------------
            Predictions_test, PtdLabels_test, Performance_train, Performance_test = NB.naive_bayes_classification(PARAMS, data_dict)
            kwargs = {
                    '0':'Accuracy:'+str(Performance_test[0]),
                    '1':'F_score_mu:'+str(Performance_test[1]),
                    '2':'F_score_sp:'+str(Performance_test[2]),
                    '3':'F_score_avg:'+str(Performance_test[3]),
                    '4':'featName:'+str(PARAMS['featName']),
                    }
            misc.print_results(PARAMS, **kwargs)
                # -------------------------------------------------------------
        # ---------------------------------------------------------------------



        # ---------------------------------------------------------------------
        ### -------------------------------- DNN ------------------------------
        # ---------------------------------------------------------------------
        elif PARAMS['clFunc']=='DNN':
            import lib.dnn_classifier as DNN

            Train_Params = DNN.train_dnn(PARAMS, data_dict)
            Test_Params = DNN.test_dnn(PARAMS, data_dict, Train_Params)

            print('Test accuracy=', Test_Params['fscore'])

            kwargs = {
                    '0':'epochs:'+str(Train_Params['epochs']),
                    '1':'batch_size:'+str(Train_Params['batch_size']),
                    '2':'learning_rate:'+str(Train_Params['learning_rate']),
                    '3':'training_time:'+str(Train_Params['trainingTimeTaken']),
                    '4':'loss:'+str(Test_Params['loss']),
                    '5':'performance:'+str(Test_Params['performance']),
                    '6':'F_score_mu:'+str(Test_Params['fscore'][0]),
                    '7':'F_score_sp:'+str(Test_Params['fscore'][1]),
                    '8':'F_score_avg:'+str(Test_Params['fscore'][2]),
                    }
            misc.print_results(PARAMS, **kwargs)
            Train_Params = None
            Test_Params = None
        # ---------------------------------------------------------------------



        # ---------------------------------------------------------------------
        ### -------------------------------- CNN ------------------------------
        # ---------------------------------------------------------------------
        elif PARAMS['clFunc']=='CNN':
            import lib.cnn_classifier as CNN

            Train_Params = CNN.train_cnn(PARAMS, data_dict)
            Test_Params = CNN.test_cnn(PARAMS, data_dict, Train_Params)

            print('Test accuracy=', Test_Params['fscore'])

            kwargs = {
                    '0':'epochs:'+str(Train_Params['epochs']),
                    '1':'batch_size:'+str(Train_Params['batch_size']),
                    '2':'learning_rate:'+str(Train_Params['learning_rate']),
                    '3':'training_time:'+str(Train_Params['trainingTimeTaken']),
                    '4':'loss:'+str(Test_Params['loss']),
                    '5':'performance:'+str(Test_Params['performance']),
                    '6':'F_score_mu:'+str(Test_Params['fscore'][0]),
                    '7':'F_score_sp:'+str(Test_Params['fscore'][1]),
                    '8':'F_score_avg:'+str(Test_Params['fscore'][2]),
                    }
            misc.print_results(PARAMS, **kwargs)
            Train_Params = None
            Test_Params = None
        # ---------------------------------------------------------------------



        if PARAMS['use_GPU']:
            reset_GPU_session()

