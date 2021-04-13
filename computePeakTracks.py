#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:38:10 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import lib.spectral_peak_tracking as SPT
import datetime
import os
import librosa.display
import random
import time
from scipy.io import wavfile
import lib.audio_processing as aproc
import configparser
import json





def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['computePeakTracks.py']
    PARAMS = {
            'Suffix': 'PF',
            'Tw': int(section['Tw']), # Frame size in miliseconds
            'Ts': int(section['Ts']), # Frame shift in miliseconds
            'nTopPeaks': int(section['nTopPeaks']), # Number of Prominent peaks to be considered for SPS computation
            'silThresh': float(section['silThresh']),  # Silence Threshold
            'intervalSize': int(section['intervalSize']), # In miliseconds
            'intervalShift': int(section['intervalShift']), # In miliseconds
            'audio_path': section['audio_path'],
            'output_path': section['output_path'],
            'dataset_name': section['dataset_name'],
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'opDir':'',
            }

    PARAMS['opDir'] = PARAMS['output_path'] + PARAMS['Suffix'] + '/' + PARAMS['dataset_name']+'/'+ str(PARAMS['intervalSize']) + 'ms_' + str(PARAMS['Tw'])+'ms_'+str(PARAMS['Ts']).replace('.','-')+'ms_'+ str(PARAMS['nTopPeaks']) +'PT_' + PARAMS['today']
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    with open(PARAMS['opDir'] + '/configuration.txt', 'a+', encoding = 'utf-8') as file:
        file.write(json.dumps(PARAMS))
    
    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    print(PARAMS['today'])

    opDir_ASPT = PARAMS['opDir'] + '/CBoW-ASPT/'
    if not os.path.exists(opDir_ASPT):
        os.makedirs(opDir_ASPT)
        
    opDir_LSPT = PARAMS['opDir'] + '/CBoW-LSPT/'
    if not os.path.exists(opDir_LSPT):
        os.makedirs(opDir_LSPT)
        
    opDirMSD_ASPT = PARAMS['opDir'] + '/MSD-ASPT/'
    if not os.path.exists(opDirMSD_ASPT):
        os.makedirs(opDirMSD_ASPT)
        
    opDirMSD_LSPT = PARAMS['opDir'] + '/MSD-LSPT/'
    if not os.path.exists(opDirMSD_LSPT):
        os.makedirs(opDirMSD_LSPT)

    opDirMSD_ASPT_LSPT = PARAMS['opDir'] + '/MSD-ASPT-LSPT/'
    if not os.path.exists(opDirMSD_ASPT_LSPT):
        os.makedirs(opDirMSD_ASPT_LSPT)

    DIR = next(os.walk(PARAMS['audio_path']))[1]
    DIR_temp = []
    for dir_i in DIR:
        print(dir_i)
        if not dir_i.startswith('__') and not dir_i.startswith('.'):
            DIR_temp.append(dir_i)
    DIR = DIR_temp
    
    if PARAMS['dataset_name']=='musan':
        DIR = [DIR[1], DIR[-1]] # music, speech
    
    clNum = -1
    for fold in DIR:
        clNum += 1
        
        path = PARAMS['audio_path']+'/'+fold 	#path where the audio is
        print('\n\n',path)
        files = librosa.util.find_files(path, ext=["wav"])
        numFiles = np.size(files)
        
        randFileIdx = list(range(np.size(files)))
        random.shuffle(randFileIdx)
        
        opDir_ASPT_Fold = opDir_ASPT + fold + '/'
        if not os.path.exists(opDir_ASPT_Fold):
            os.makedirs(opDir_ASPT_Fold)
            
        opDir_LSPT_Fold = opDir_LSPT + fold + '/'
        if not os.path.exists(opDir_LSPT_Fold):
            os.makedirs(opDir_LSPT_Fold)
            
        opDirMSD_ASPT_Fold = opDirMSD_ASPT + fold + '/'
        if not os.path.exists(opDirMSD_ASPT_Fold):
            os.makedirs(opDirMSD_ASPT_Fold)
            
        opDirMSD_LSPT_Fold = opDirMSD_LSPT + fold + '/'
        if not os.path.exists(opDirMSD_LSPT_Fold):
            os.makedirs(opDirMSD_LSPT_Fold)

        opDirMSD_ASPT_LSPT_Fold = opDirMSD_ASPT_LSPT + fold + '/'
        if not os.path.exists(opDirMSD_ASPT_LSPT_Fold):
            os.makedirs(opDirMSD_ASPT_LSPT_Fold)

        fileCount = 0 #to see in the terminal how many files have been loaded
        for f in range(numFiles):
            audio = files[randFileIdx[f]]
            fileCount += 1
            print('\n', fileCount, ' Audio file: ', audio, fold)
            
            check_file = opDir_ASPT_Fold + audio.split('/')[-1].split('.')[0] + '.npy'
            print('Check file: ', check_file)
            if os.path.exists(check_file):
                print('Feature file already exists!!!\n\n\n')
                continue
            
            
            fs, Xin_readonly = wavfile.read(audio)
            Xin = Xin_readonly.copy()
            print('Sampling rate=', fs)
            
            # Removing the silences
            Xin_silrem, silPos = aproc.removeSilence(Xin=Xin, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], fs=fs, threshold=PARAMS['silThresh'])
            print('File duration: ', np.round(len(Xin_silrem)/fs,2),' (', np.round(len(Xin)/fs,2), ')')
            if np.size(Xin_silrem)<=1:
                continue;
            Xin = Xin_silrem
            if len(Xin)/fs < 1:
                continue

            frameSize = int((PARAMS['Tw']*fs)/1000) # Frame size in number of samples
            frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples

            '''
            Creating the corresponding SPT matrix
            '''
            start = time.process_time()
            
            LSPT, ASPT, stft, peak_repeat_count = SPT.computeSPT(y=Xin, sr=fs, frame_length=frameSize, hop_length=frameShift, req_pks=PARAMS['nTopPeaks'])
            
            print('Time for SPT: ',np.round(time.process_time()-start,2),'s')
            
            numSamplesInInterval = int(PARAMS['intervalSize']*fs/1000)
            numSamplesShiftPerInterval = int(PARAMS['intervalShift']*fs/1000)
            numFv = int(np.floor((len(Xin)-numSamplesInInterval)/numSamplesShiftPerInterval))+1
            nFrames = np.shape(ASPT)[0]

            peak_repeat_statistics_file = PARAMS['opDir'] + '/' + fold + '_peak_repeat_statistics.csv'
            fid = open(peak_repeat_statistics_file, 'a+', encoding='utf8')
            fid.write(audio + '\t' + str(peak_repeat_count) + '\t' + str(nFrames) + '\n')
            fid.close()
            
            if numFv==0:
                continue
            nmFrmPerInterval = int(np.floor((numSamplesInInterval-frameSize)/frameShift))+1
            nmFrmPerIntervalShift = int(np.floor((numSamplesShiftPerInterval-frameSize)/frameShift))+1
            print('Signal: ', len(Xin), numSamplesInInterval, numFv, nFrames, nmFrmPerInterval, np.shape(ASPT))
            
            frmStart = 0
            frmEnd = 0
            fv_val = np.array
            fv_loc = np.array
            for l in range(numFv):
                frmStart = l*nmFrmPerIntervalShift
                frmEnd = l*nmFrmPerIntervalShift + nmFrmPerInterval
                if frmEnd>nFrames:
                    frmEnd = nFrames
                    frmStart = frmEnd-nmFrmPerInterval
                val = np.array(ASPT[frmStart:frmEnd, :], ndmin=2)
                val = np.expand_dims(val, axis=0)
                loc = np.array(LSPT[frmStart:frmEnd, :], ndmin=2)
                loc = np.expand_dims(loc, axis=0)
                if np.size(fv_val)<=1:
                    fv_val = val
                    fv_loc = loc
                else:
                    fv_val = np.append(fv_val, val, 0)
                    fv_loc = np.append(fv_loc, loc, 0)

            fileName_val = opDir_ASPT_Fold + audio.split('/')[-1].split('.')[0] + '.npy'
            np.save(fileName_val, fv_val)

            fileName_loc = opDir_LSPT_Fold + audio.split('/')[-1].split('.')[0] + '.npy'
            np.save(fileName_loc, fv_loc)

            print('APT: ', np.shape(fv_val), ' LPT: ', np.shape(fv_loc), ' Sampling rate: ', fs)

            start = time.process_time()
            MSD_LSPT = SPT.computeSPT_MeanStd(numFv, nmFrmPerInterval, nmFrmPerIntervalShift, nFrames, LSPT)
            MSD_LSPT = np.array(MSD_LSPT, ndmin=2)
            print('MSD_LSPT: ', np.shape(MSD_LSPT))
            fileName = opDirMSD_LSPT_Fold + audio.split('/')[-1].split('.')[0] + '.npy'
            np.save(fileName, MSD_LSPT)

            MSD_ASPT = SPT.computeSPT_MeanStd(numFv, nmFrmPerInterval, nmFrmPerIntervalShift, nFrames, ASPT)
            MSD_ASPT = np.array(MSD_ASPT, ndmin=2)
            print('MSD_ASPT: ', np.shape(MSD_ASPT))
            fileName = opDirMSD_ASPT_Fold + audio.split('/')[-1].split('.')[0] + '.npy'
            np.save(fileName, MSD_ASPT)

            MSD_ASPT_LSPT = np.append(MSD_ASPT, MSD_LSPT, 1)
            MSD_ASPT_LSPT = np.array(MSD_ASPT_LSPT, ndmin=2)
            print('MSD_ASPT_LSPT: ', np.shape(MSD_ASPT_LSPT))
            fileName = opDirMSD_ASPT_LSPT_Fold + audio.split('/')[-1].split('.')[0] + '.npy'
            np.save(fileName, MSD_ASPT_LSPT)

            print('Time for MSD Features: ',np.round(time.process_time() - start,2),'s')
