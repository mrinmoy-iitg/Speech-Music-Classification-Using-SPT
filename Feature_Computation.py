#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:11:00 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import datetime
import os
import librosa.display
import random
import glob
import lib.audio_processing as aud_proc
import configparser
from scipy import interpolate



def compute_MFCC_feature(PARAMS, Xin, fs):
    frameSize = int((PARAMS['Tw']*fs)/1000) # Frame size in number of samples
    frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples
    hann_window = np.hanning(frameSize)
    MFCC = librosa.feature.mfcc(
            y=Xin, 
            sr=fs, 
            n_mfcc=PARAMS['nMFCC'], 
            norm='ortho', 
            n_fft=frameSize, 
            hop_length=frameShift,
            window=hann_window,
            )
    MFCC = np.transpose(MFCC)
    print('MFCC shape=',np.shape(MFCC))
    
    FV = np.array
    for i in range(0,np.shape(MFCC)[0]-50,50):
        if np.size(FV)<=1:
            FV = np.array(np.reshape(MFCC[i:i+50,:], 2000), ndmin=2)
        else:
            FV = np.append(FV, np.array(np.reshape(MFCC[i:i+50,:], 2000), ndmin=2), 0)
        
    return FV

    



def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Feature_Computation.py']
    PARAMS = {
            'data_folder': section['data_folder'],
            'output_folder': section['output_folder'],
            'featType': section['featType'], # EF, PF, IM
            'Tw': int(section['Tw']), # frame size in miliseconds
            'Ts': int(section['Ts']), # frame shift in miliseconds
            'nMFCC': int(section['nMFCC']),
            'n_mels': int(section['n_mels']),
            'silThresh': float(section['silThresh']),
            'intervalSize': int(section['intervalSize']), # interval size in miliseconds
            'intervalShift': int(section['intervalShift']), # interval shift in miliseconds
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            }

    return PARAMS



def resize_spectrogram(fv, new_size):
    x = np.arange(0,np.shape(fv)[1],1)
    y = np.arange(0,np.shape(fv)[0],1)
    interpolation_func = interpolate.interp2d(x, y, fv, kind='linear')
    xnew = np.linspace(0,np.shape(fv)[1], new_size[1])
    ynew = np.linspace(0,np.shape(fv)[0], new_size[0])
    fv_interpolated = interpolation_func(xnew, ynew)
    
    return fv_interpolated
    



if __name__ == '__main__':

    PARAMS = __init__()
    print(PARAMS['today'])
    
    k = -1
    dataset_name = PARAMS['data_folder'].split("/")[k]
    while dataset_name=='':
        k -= 1
        dataset_name = PARAMS['data_folder'].split("/")[k]
        
    opDir = PARAMS['output_folder'] + '/' + PARAMS['featType'] + '/' + dataset_name + '/' + str(PARAMS['intervalSize']) + 'ms_' + str(PARAMS['Tw']) + 'ms_' + str(PARAMS['Ts']).replace('.','-') + 'ms_' + PARAMS['today'] + '/'    
    
    if not os.path.exists(opDir):
        os.makedirs(opDir)
    
    All_Data = np.array
    All_Label = np.array
    
	#searching through all the folders and picking up audio
    DIR = next(os.walk(PARAMS['data_folder']))[1]
    if dataset_name=='musan':
        DIR = [DIR[1], DIR[-1]] # Music and Speech
    
    if dataset_name=='Scheirer-slaney':
        DIR = [DIR[1], DIR[-1]] # Music and Speech
    
    
    clNum = -1
    for fold in DIR:
        clNum += 1
        
        path = PARAMS['data_folder'] + '/' + fold 	#path where the audio is
        print('\n\n\n',path)
        files = librosa.util.find_files(path, ext=["wav"])
        numFiles = np.size(files)
        
        randFileIdx = list(range(np.size(files)))
        random.shuffle(randFileIdx)
        count = 1	#to see in the terminal how many files have been loaded
        
        opDirFold = opDir + '/' + fold + '/'
        if not os.path.exists(opDirFold):
            os.makedirs(opDirFold)

        for f in range(numFiles): #numFiles
            audio = files[randFileIdx[f]]
            fName = opDirFold + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
            print(count,fold,fName)
            
            matchingFiles = glob.glob(fName)
            print('Matching files: ', len(matchingFiles))
            if len(matchingFiles)>0:
                print('\n\n\nFile exists!!!\t',matchingFiles,'\n\n\n')
                count += 1
                continue                
            
            print('\n\n\nAudio file: ', audio)
            Xin, fs = librosa.core.load(audio, mono=True, sr=None)
            # Removing the silences
            Xin_silrem, silPos = aud_proc.removeSilence(Xin=Xin, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], fs=fs, threshold=PARAMS['silThresh'])
            if np.size(Xin_silrem)<=1:
                continue
                
            print('File duration: ', np.round(len(Xin_silrem)/fs,2),' (', np.round(len(Xin)/fs,2), ')')
            Xin = Xin_silrem
            if len(Xin)/fs < 1:
                continue
            
            frameSize = int((PARAMS['Tw']*fs)/1000) # Frame size in number of samples
            frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples
            signalLength = len(Xin)*1000/fs
            numIntervals = int(np.floor((signalLength - PARAMS['intervalSize'])/PARAMS['intervalShift']) + 1)
            framesPerInterval = int((PARAMS['intervalSize']-PARAMS['Tw'])/PARAMS['Ts'] + 1)
            framesShiftPerInterval = int((PARAMS['intervalShift']-PARAMS['Tw'])/PARAMS['Ts'] + 1)
            

            if PARAMS['featType']=='IM':
                FV = np.empty([])
                MelSpec = librosa.feature.melspectrogram(
                        y=Xin, 
                        sr=fs,
                        n_mels=PARAMS['n_mels'], #**kwargs
                        n_fft=frameSize, 
                        hop_length=frameShift)
                MelSpec = MelSpec[-1::-1,:].astype(np.float32)
                count = 0
                frmStart = -framesShiftPerInterval
                frmEnd = framesPerInterval - framesShiftPerInterval
                for l in range(numIntervals):
                    frmStart = frmStart + framesShiftPerInterval
                    frmEnd = np.min([frmEnd+framesShiftPerInterval, np.shape(MelSpec)[1]])
                    if (frmEnd-frmStart)<framesPerInterval:
                        frmStart = frmEnd - framesPerInterval
                    spec_interval = MelSpec[:,frmStart:frmEnd]
                    fv = np.array(spec_interval, ndmin=2)
                    fv_resized = resize_spectrogram(fv, [200,200])
                    fv_resized = np.array(fv_resized, ndmin=3).astype(np.float32)
                    fileName = fName + str(f) + '_' + str(count) + '.npy'
                    np.save(fileName, fv_resized)
                    count += 1
                    print('\t\t\tInterval ', l, ': ', fv_resized.shape)
                

            elif PARAMS['featType']=='MFCC':
                
                FV = compute_MFCC_feature(PARAMS, Xin, fs)
                
                if np.size(All_Data)<=1:
                    All_Data = np.array(FV, ndmin=2)
                    All_Label = np.ones(np.shape(FV)[0])*clNum
                else:
                    All_Data = np.append(All_Data, np.array(FV, ndmin=2), 0)
                    All_Label = np.append(All_Label, np.ones(np.shape(FV)[0])*clNum)
            
                
                '''
                Saving features as one file
                '''
                fileName = opDirFold + audio.split('/')[-1].split('.')[0] + '.npy'
                np.save(fileName, FV)


            elif PARAMS['featType']=='CFA':
                print('Audio file: ', audio, '\n')
                Xin_cfa, fs_cfa = librosa.core.load(audio, mono=True, sr=11025)
                print('Sampling rate: ', fs_cfa)
                # Removing the silences
                Xin_cfa, silPos = aud_proc.removeSilence(Xin=Xin_cfa, Tw=46, Ts=23, fs=fs_cfa, threshold=PARAMS['silThresh'])
                if np.size(Xin_cfa)<=1:
                    continue;
                print('File duration: ', np.round(len(Xin_cfa)/fs_cfa,2),' (', np.round(len(Xin_cfa)/fs_cfa,2), ')')
                if len(Xin_cfa)/fs < 1:
                    continue
                frameSize = 512 # Frame size in number of samples
                frameShift = 256 # Frame shift in number of samples
                
                X = librosa.core.stft(y=Xin, n_fft=frameSize, hop_length=frameShift)
                X = np.power(np.abs(X),2)

                intervalSize = PARAMS['intervalSize'] #msec
                numSamplesInInterval = int(intervalSize*fs/1000)
                numFv = int(np.floor(np.size(Xin)/numSamplesInInterval))
                nFrames = np.shape(X)[1]
                if numFv==0:
                    continue
                nmFrmPerInterval = int(np.round(nFrames/numFv))

                FV = np.empty(shape=(numFv,20))
                frmStart = -1
                frmEnd = 0
                for l in range(numFv):
                    fv = np.array
                    frmStart = frmEnd + 1;
                    frmEnd = np.min([frmEnd + nmFrmPerInterval, nFrames])

                    CX_interval = X[:,frmStart:frmEnd]
                    theta_s = np.median(CX_interval)
                    # Clipped Power Spectrogram
                    CX_interval = CX_interval > theta_s
                    CX_interval = CX_interval*1
                    # Selecting frequency bins upto 2.15 KHz
                    CX_interval = CX_interval[:100,:]

                    SDF = np.empty(shape=(np.shape(CX_interval)[0], 1))
                    for k in range(np.shape(CX_interval)[0]):
                        sdf = 0
                        for t in range(1,np.shape(CX_interval)[1]-1):
                            sdf += CX_interval[k,t-1]*CX_interval[k,t]*CX_interval[k,t+1]
                        SDF[k] = sdf
                    # Normalizing with number of frames
                    SDF /= np.shape(CX_interval)[1]
                    groupCount = 0
                    for i in range(0,100,5):
                        FV[l,groupCount] = np.mean(SDF[i:i+5,0])
                        groupCount += 1
                    
                FV = np.array(FV, ndmin=2)
                print('FV shape: ', np.shape(FV))
                fileName = opDirFold + audio.split('/')[-1].split('.')[0] + '.npy'
                np.save(fileName, FV)



            elif PARAMS['featType']=='BN': 
                '''
                Bottleneck feature computation input, 13 dimensional MFCCs over on interval concatenated as one Feature Vector
                '''
                frameSize = int((PARAMS['Tw']*fs)/1000) # Frame size in number of samples
                frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples
                hann_window = np.hanning(frameSize)
                intervalSizeSamples = int(PARAMS['intervalSize']*fs/1000)
                intervalShiftSamples = int(PARAMS['intervalShift']*fs/1000)
                numIntervals = int((len(Xin)-intervalSizeSamples)/intervalShiftSamples)+1
                smpStart = 0
                smpEnd = 0
                FV = np.empty([])
                for intvl in range(numIntervals):
                    smpStart = intvl*intervalShiftSamples
                    smpEnd = intvl*intervalShiftSamples + intervalSizeSamples
                    if smpEnd>len(Xin):
                        smpEnd = len(Xin)
                        smpStart = smpEnd-intervalSizeSamples
                    Xin_interval = Xin[smpStart:smpEnd]
                    MFCC = librosa.feature.mfcc(y=Xin_interval, sr=fs, n_mfcc=PARAMS['nMFCC'], n_fft=frameSize, hop_length=frameShift, window=hann_window)
                    FV_intvl = np.array(np.transpose(MFCC).flatten(), ndmin=2)
                    if intvl==0:
                        FV = FV_intvl
                    else:
                        FV = np.append(FV, FV_intvl, 0)

                '''
                Saving features of one file
                '''
                fileName = opDirFold + audio.split('/')[-1].split('.')[0] + '.npy'
                np.save(fileName, FV)
                print('FV shape=',np.shape(FV), ' numIntervals=', numIntervals)
                FV = None
                del FV

            count += 1