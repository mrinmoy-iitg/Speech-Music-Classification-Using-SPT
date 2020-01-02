#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:46:52 2019

@author: mrinmoy
"""
import numpy as np
import librosa
import os
import scipy.io.wavfile as wavIO
import lib.audio_processing as aproc


def dbToRatio(dB):
    ratio = np.power(10, (np.array(dB)/20))
    return ratio


if __name__ == '__main__':
    folder = '/media/mrinmoy/NewVolume/PhD_Work/Data/'
    
    path1 = folder + 'GTZAN'
    path2 = folder + 'GTZAN_Mixed_dB'
    
    print(path1)
    print(path2)
    
    dB = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
    ratios = dbToRatio(dB)
    
    Tw = 10 # in msec
    Ts = 5 # in msec
    silThresh = 0.05
    
    classes = {0:'music', 1:'speech'}
    classes_mixed = {
            0:'mixture_-20dB',
            1:'mixture_-10dB',
            2:'mixture_-5dB',
            3:'mixture_-2dB',
            4:'mixture_-1dB',
            5:'mixture_0dB',
            6:'mixture_1dB',
            7:'mixture_2dB',
            8:'mixture_5dB',
            9:'mixture_10dB',
            10:'mixture_20dB',
            }
    
    for mixRatio in range(len(classes_mixed)):
        opDir = path2+'/'+classes_mixed[mixRatio]+'/'
        if not os.path.exists(opDir):
            os.makedirs(opDir)
        print('opDir: ', opDir)
    
        music_filepath = path1+'/'+classes[0] 	#path where the audio is
        speech_filepath = path1+'/'+classes[1] 	#path where the audio is
        
        music_files = librosa.util.find_files(music_filepath, ext=["wav"])
        np.random.shuffle(music_files)
        speech_files = librosa.util.find_files(speech_filepath, ext=["wav"])
        np.random.shuffle(speech_files)
    
        print('Length of file list: ', len(music_files))
        i = 0
        for i in range(64):
            randFileIdx1 = np.squeeze(np.random.randint(0, len(music_files), 1))+0
            randFileIdx2 = np.squeeze(np.random.randint(0, len(speech_files), 1))+0
            
            music_audio = music_files[randFileIdx1]
            print(music_audio)
            speech_audio = speech_files[randFileIdx2]
            print(speech_audio)

            Xin_music, fs = librosa.core.load(music_audio, mono=True, sr=None)
            Xin_music, silPos1 = aproc.removeSilence(Xin_music, Tw, Ts, fs, silThresh)
            
            Xin_speech, fs = librosa.core.load(speech_audio, mono=True, sr=None)
            Xin_speech, silPos2 = aproc.removeSilence(Xin_speech, Tw, Ts, fs, silThresh)
            
            minSize = np.min([np.size(Xin_music), np.size(Xin_speech)])

            Xin_music -= np.mean(Xin_music)
            Xin_music /= np.max(np.abs(Xin_music))
            Xin_speech -= np.mean(Xin_speech)
            Xin_speech /= np.max(np.abs(Xin_speech))


            mixedXin = Xin_music[:minSize] + Xin_speech[:minSize]*ratios[mixRatio]

            mixedXin -= np.mean(mixedXin)
            mixedXin /= np.max(np.abs(mixedXin))
            
            
            fName = opDir + '/' + str(i) + '.wav'
            wavIO.write(filename=fName, rate=fs, data=mixedXin)
            
            print('File created: ', fName)
                
            i += 1
