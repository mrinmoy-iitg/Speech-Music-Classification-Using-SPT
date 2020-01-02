#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:34:47 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import configparser
import datetime
import numpy as np
import lib.misc as misc
import lib.dnn_classifier as DNN
import lib.svm_classifier as SVM
import lib.gmm_classifier as GMM
import lib.NB_classifier as NB


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



'''
Initialize the script
'''
def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Ensemble_Classifier.py']
    PARAMS = {
            'folder': section['folder'],
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'DIM': [],
            'command': 'Classify',
            'featName': section['featName'],
            'clFunc': section['clFunc'], #DNN-Ensemble, SVM-Ensemble
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
            'generalization_perf': float(section['generalization_perf']),
            'feat_indexes':{
                     'CBoW-ASPT':list(range(0, 100)),
                     'CBoW-LSPT':list(range(100, 200)),
                    # ---------------------------------------------------------
                    # 'MSD-ASPT':list(range(0, 20)),
                    # 'MSD-LSPT':list(range(20, 40)),
                    # ---------------------------------------------------------
                    },
            }
    PARAMS['DIM'] = list(range(PARAMS['dim_range1'], PARAMS['dim_range2']))
   
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
        print('opDir: ', PARAMS['opDir'])
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
        misc.print_congiguration(PARAMS)

        if not PARAMS['data_generator']:
            data_dict = misc.get_data(PARAMS)
        else:
            data_dict = {'train_data':np.empty, 'train_label':np.empty, 'val_data':np.empty, 'val_label':np.empty, 'test_data':np.empty, 'test_label':np.empty}
            
        Majority_Voting_Ensemble_Result = np.zeros((np.shape(data_dict['test_data'])[0], len(PARAMS['classes'])))
        All_Classifier_Predictions = np.array
        
        num_classifiers = len(PARAMS['feat_indexes'])
            
        feature_indexes_dict_keys = [key for key in PARAMS['feat_indexes'].keys()]
        for classifier_num in range(num_classifiers):
            print('\n\n\nClassifier num ', classifier_num+1)
            PARAMS['DIM'] = list(range(PARAMS['dim_range1'], PARAMS['dim_range2']))

            if PARAMS['use_GPU']:
                PARAMS['GPU_session'] = start_GPU_session()

            data_dict_Classifier_Part = data_dict.copy()
            
            if not PARAMS['data_generator']:
                DIM = PARAMS['feat_indexes'][feature_indexes_dict_keys[classifier_num]]
                print('DIM: ', DIM)
                
                data_dict_Classifier_Part['train_data'] = data_dict['train_data'][:, DIM]
                data_dict_Classifier_Part['val_data'] = data_dict['val_data'][:, DIM]
                data_dict_Classifier_Part['test_data'] = data_dict['test_data'][:, DIM]
                
                if PARAMS['PCA_flag']:
                    data_dict_Classifier_Part['train_data'], data_dict_Classifier_Part['val_data'], data_dict_Classifier_Part['test_data'] = misc.pca_dim_reduction(data_dict_Classifier_Part['train_data'], data_dict_Classifier_Part['val_data'], data_dict_Classifier_Part['test_data'])
            else:
                PARAMS['DIM'] = PARAMS['feat_indexes'][feature_indexes_dict_keys[classifier_num]]
                
            PARAMS['modelName'] = PARAMS['opDir'] + '/'+ PARAMS['clFunc'] +'_Custom_Classifier_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(numIter) + '.xyz'
            
            
            
            
            if PARAMS['clFunc']=='DNN-Ensemble':
                train_params_file = PARAMS['opDir'] + '/train_params_Classifier_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(numIter) + '.pkl'
                if not os.path.exists(train_params_file):
                    Train_Params = DNN.train_dnn(PARAMS, data_dict_Classifier_Part)
                    if PARAMS['save_flag']:
                        misc.save_obj(Train_Params, PARAMS['opDir'], 'train_params_Classifier_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(numIter))
                else:
                    Train_Params = misc.load_obj(PARAMS['opDir'], 'train_params_Classifier_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(numIter))
    

                test_params_file = PARAMS['opDir'] + '/test_params_Classifier_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(numIter) + '.pkl'
                if not os.path.exists(test_params_file):
                    Test_Params = DNN.test_dnn(PARAMS, data_dict_Classifier_Part, Train_Params)
                    if PARAMS['save_flag']:
                        misc.save_obj(Test_Params, PARAMS['opDir'], 'test_params_Classifier_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(numIter))
                else:
                    Test_Params = misc.load_obj(PARAMS['opDir'], 'test_params_Classifier_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(numIter))
        
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
                print('Performance: ', Test_Params['performance'], Test_Params['ConfMat'])
                print('Avg. F1-score: ', Test_Params['fscore'][-1])
                print('Training time taken: ', Train_Params['trainingTimeTaken'], Test_Params['testingTimeTaken'])





            elif PARAMS['clFunc']=='SVM-Ensemble':
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                PARAMS['modelName'] = PARAMS['opDir'] + '/SVM_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(PARAMS['iter']) + '.pkl'
                Train_Params, Test_Params = SVM.grid_search_svm(PARAMS, data_dict_Classifier_Part)
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
                print('Performance: ', Test_Params['Performance_test'], Test_Params['accuracy_test'])
                print('Training time taken: ', Train_Params['trainingTimeTaken'], Test_Params['testingTimeTaken'])
                



            elif PARAMS['clFunc']=='GMM-Ensemble':
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                PARAMS['modelName'] = PARAMS['opDir'] + '/GMM_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(PARAMS['iter']) + '.pkl'
                # Performing Grid-Search --------------------------------------
                score_test, PtdLabels_test, K, Performance_train, Performance_test = GMM.grid_search_gmm(PARAMS, data_dict_Classifier_Part)
                kwargs = {
                        '0':'K:'+str(K),
                        '1':'Accuracy:'+str(Performance_test[0]),
                        '2':'F_score_mu:'+str(Performance_test[1]),
                        '3':'F_score_sp:'+str(Performance_test[2]),
                        '4':'F_score_avg:'+str(Performance_test[3]),
                        '5':'featName:'+str(PARAMS['featName']),
                        }
                misc.print_results(PARAMS, **kwargs)
                Test_Params = {
                    'Predictions_test': score_test,
                    'PtdLabels_test': PtdLabels_test,
                    }


            elif PARAMS['clFunc']=='NB-Ensemble':
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                PARAMS['modelName'] = PARAMS['opDir'] + '/NB_' + feature_indexes_dict_keys[classifier_num] + '_iter' + str(PARAMS['iter']) + '.pkl'
                # Performing Grid-Search --------------------------------------
                Predictions_test, PtdLabels_test, Performance_train, Performance_test = NB.naive_bayes_classification(PARAMS, data_dict_Classifier_Part)
                kwargs = {
                        '0':'Accuracy:'+str(Performance_test[0]),
                        '1':'F_score_mu:'+str(Performance_test[1]),
                        '2':'F_score_sp:'+str(Performance_test[2]),
                        '3':'F_score_avg:'+str(Performance_test[3]),
                        '4':'featName:'+str(PARAMS['featName']),
                        }
                misc.print_results(PARAMS, **kwargs)
                Test_Params = {
                    'Predictions_test': Predictions_test,
                    'PtdLabels_test': PtdLabels_test,
                    }

    
            ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
            Majority_Voting_Ensemble_Result += Test_Params['Predictions_test']
            if np.size(All_Classifier_Predictions)<=1:
                All_Classifier_Predictions = np.array(Test_Params['PtdLabels_test'], ndmin=2)
            else:
                All_Classifier_Predictions = np.append(All_Classifier_Predictions, np.array(Test_Params['PtdLabels_test'], ndmin=2), 0)
    
            data_dict_Classifier_Part = None
            Train_Params = None
            Test_Params = None
            del data_dict_Classifier_Part
            del Train_Params
            del Test_Params

            if PARAMS['use_GPU']:
                reset_GPU_session()

        PtdLabels_majority_voting = np.argmax(Majority_Voting_Ensemble_Result, axis=1)
        ConfMat_majority_voting, fscore_majority_voting = misc.getPerformance(PtdLabels_majority_voting, data_dict['test_label'])
        print('\n\n\nMajority Voting Ensemble Avg. F1-score: ', np.mean(fscore_majority_voting))
        
        resultFile = PARAMS['opDir'] + '/Ensemble_performance_' + PARAMS['featName'] + '.csv'
        result_fid = open(resultFile, 'a+', encoding='utf-8')
        # result_fid.write('Majority Voting Ensemble Average=' + str(np.round(fscore_majority_voting[-1],4)) + ' F1-score= ' + str([str(np.round(fscore_majority_voting[i], 2)) for i in range(len(fscore_majority_voting)-1)]) + '\n')
        result_fid.write('Majority Voting Ensemble Average\t' + str(fscore_majority_voting[0]) + '\t'  + str(fscore_majority_voting[1]) + '\t' + str(fscore_majority_voting[2]) + '\n')
        result_fid.close()
    

        kwargs = {
                '0':':',
                '1':':',
                '2':':',
                '3':':',
                '4':':',
                '5':':Majority Voting Ensemble',
                '6':'F_score_mu:'+str(fscore_majority_voting[0]),
                '7':'F_score_sp:'+str(fscore_majority_voting[1]),
                '8':'F_score_avg:'+str(fscore_majority_voting[2]),
                }
        misc.print_results(PARAMS, **kwargs)
    
    
