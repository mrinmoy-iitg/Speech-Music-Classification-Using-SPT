#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:28:27 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import numpy as np
import librosa



def removeSilence(Xin, Tw, Ts, fs, threshold):
    Xin = Xin - np.mean(Xin)
    Xin = Xin / np.max(np.abs(Xin))
    
    frameSize = int((Tw*fs)/1000) # Frame size in number of samples
    frameShift = int((Ts*fs)/1000) # Frame shift in number of samples
    
    Rmse = librosa.feature.rms(y=Xin, frame_length=frameSize, hop_length=frameShift)
    energy = Rmse[0,:] #pow(Rmse,2)
    energyThresh = threshold*np.max(energy) # TEST WITH MEAN

    silMarker = energy
    silMarker[silMarker < energyThresh] = 0
    silMarker[silMarker >= energyThresh] = 1
    silences = np.empty([])
    totalSilDuration = 0.005

    # Suppressing spurious noises -----------------------------------------        
    winSz = 3 # Giving a window size of 110 ms
    i = winSz
    while i < len(silMarker)-winSz:
        if np.sum(silMarker[i-winSz:i+winSz]==1) <= np.ceil(2*winSz*0.3):
            silMarker[i] = 0
        i = i + 1
    # ---------------------------------------------------------------------
    
    silPos = np.zeros([len(Xin)])
    shiftSamples = round(Ts*fs/1000)
    
    i=0
    while i<len(silMarker):
        while silMarker[i]==0:
            if i == len(silMarker)-1:
                break
            i = i + 1
        j = i
        while silMarker[j]==1:
            if j == len(silMarker)-1:
                break
            j = j + 1
        k = np.max([shiftSamples*(i-1),1])
        l = np.min([shiftSamples*(j-1),len(Xin)]);
        
        # Only silence segments of durations greater than given threshold
        # (e.g. 100ms) are removed
        if (l-k)/fs > threshold:
            silPos[k:l] = 1
            if np.size(silences)<=1:
                silences = np.array([k,l], ndmin=2)
            else:
                silences = np.append(silences, np.array([k,l], ndmin=2),0)
            totalSilDuration = totalSilDuration + (l-k)/fs
        i = j + 1;
    
    if np.size(silences)<=1:
        return [], []
    silLen = np.shape(silences)[0]
    Xin_silrem = np.empty([]) #Xin

    for i in range(silLen):
        Xin_silrem = np.append(Xin_silrem, Xin[silences[i,0]:silences[i,1]])
        
    return Xin_silrem, silPos




def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

