# -*- coding: utf-8 -*-

import numpy as np
from Pose_Retrival_All_class import *

def find_standing(arrayX, index_ofInterest, threshold=15):
        
    """
    It retrieves the n-th frame in which we can find the standing position. 
    To do so, it finds the frame in which the distance between the shoulder 
    and the knee is the greatest. This happens when the person is standing.

    Parameters
    ----------
    arrayX: array 
    index_ofInterest: int
        index of the body pair of interest according to whose length we want 
        to normaliz
    threshold: float
        Threshold used to find the standing frame
    Returns
    ----------
    dict_max_shoulder_knee: dict
        dict mapping each video to its frame in which the distance between 
        shoulder and knee is max
    """
        
    dict_max_shoulder_knee = {}
    for n_video in range(arrayX.shape[0]):
        
        line = arrayX[n_video][index_ofInterest][:50].copy() 

        #Add a threshold in the distance of the first frame
        #to avoid unstable result.
        line[0] += threshold 
        max_distance_frame = np.argmax(line)

        dict_max_shoulder_knee[n_video] = max_distance_frame

    return dict_max_shoulder_knee

def normalize(index_ofInterest, arrayX, threshold):
    """
    Normalize each video with the dimension of a body part. The length of the
    body part is found when the individual is standing.
    
    Parameters
    ----------
    index_ofInterest: int
        body pair of interest according to whose length we want to normalize
    arrayX: array
    threshold: float 
        Threshold used to find the standing frame (function: find_standing)
    Returns
    ----------
    array: array
    
    """
    index_ofInterest = DISTANCE[index_ofInterest]
    array = arrayX.copy()
  
    dic_stand = find_standing(arrayX, index_ofInterest, threshold=threshold)
  
    for number_video in range(len(arrayX)):
        frame_stand = dic_stand[number_video]
        body_part = arrayX[number_video][index_ofInterest][frame_stand]
        array[number_video] = array[number_video] / body_part

    return array 

def finding_minmax(arrayX_train_to_consider):
    """
    Calculate the min and max values for each body pair across all frames of 
    all videos, and return them in two separate arrays
    
    Parameters
    ----------
    arrayX_train_to_consider: 4d-array 
    
    Returns
    ----------
    array_min: array
    array_max: array
    """
    
    # comparing all videos and taking the min and the max row by row, i.e. it
    # returns two arrays - for the min and max respectively - of the body pairs
    # for each frame across videos. So, they are not global min and max yet.
    min_of_all_bodypairs = arrayX_train_to_consider[:,:,:,:].min(axis=0)
    max_of_all_bodypairs = arrayX_train_to_consider[:,:,:,:].max(axis=0)
    
    # finding the global min and max of each body pair across frames 
    array_min = min_of_all_bodypairs[:,:].min(axis=1)
    array_max = max_of_all_bodypairs[:,:].max(axis=1)
            
    return array_min, array_max


def dup_cols(a, indx, num_dups=1):
    """
    Duplicates values in a column of arrayX by adding num_dups columns. Needed 
    to be able to broadcast such array in the normalization_minmax function.
    
    Parameters
    ----------
    a: array 
    indx: int
        index representing the col to duplicate
    num_dups: int
        number of duplicates to create, that will be added next to the indx col
    Returns
    ----------
    arr: array
    """ 
    
    arr = np.insert(a,[indx+1]*num_dups,a[:,[indx]],axis=1)
    return arr


def normalization_minmax(arrayX_train_to_consider):
    """
    Applies min_max normalization to the input array. It does so for each body 
    pair, whilst min and max values are found globally
    
    Parameters
    ----------
    arrayX_train_to_consider: 4d-array
    
    Returns
    ----------
    arr: array
    """ 
    
    # find min and max value for each body pair
    min_, max_ = finding_minmax(arrayX_train_to_consider)
    
    # add filler values to get the arrays to be of the right shape
    min_ = dup_cols(min_, 0, num_dups=149)
    max_ = dup_cols(max_, 0, num_dups=149)
    min_ = min_[..., np.newaxis]
    max_ = max_[..., np.newaxis]
    
    # apply the normalization formula
    videos_normalized_2 =[]
    for i in range(len(arrayX_train_to_consider)):
        formula_normalization = (arrayX_train_to_consider[i] - min_) / (max_ - min_) 
        videos_normalized_2.append(formula_normalization)
    
    arr = np.array(videos_normalized_2)  
    return  arr