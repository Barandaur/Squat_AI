# -*- coding: utf-8 -*-

from ast import literal_eval
import os 
import pandas as pd
import numpy as np
from scipy import signal
from scipy.spatial import distance
import math
from Pose_Retrival_All_class import *

def difficult_keypoints(keypoint_matrix):
    
    """
    Given a matrix coordinate of a video, it returns how many times
    each keypoint is not detected in the video.
    
    Parameters
    ----------
    keypoint_matrix: pandas.DataFrame
    
    Returns
    -------
    df_null: pandas.Series

    """
    
    df = keypoint_matrix
    df.index = BODY_PARTS
    df_null = df.isnull()
    df_null = df_null.sum(axis=1)
    df_null.sort_values(ascending=False,inplace=True)
    
    return df_null


def fillna_coord_zeros(directory):
    """
    Fill all the null coordinates with zeros.
    It creates a list of arrays with these filled coordinates 
    ('coordinate_zero') and it collects the position that each video
    has in the above  list ('names_file').
    
    Parameters
    ----------
    directory: directory
    
    Returns
    --------
    coordinate_zero: list of arrays
        Each array is the coordinate matrix for a video where 
        the null keypoints are filled with zeros.
    names_file: dictionary
        The keys represent the position of the video in the
        coordinate_zero list, whereas the values are the 
        directory+name of the video itself.
        
    """
    
    coordinate_zero = []
    names_file = {}
    counter = 0

    for folders in sorted(os.listdir(directory)):
        DIR = directory+folders+'/'
        for file in sorted(os.listdir(DIR)):
            if not file.startswith('distance'):
                file_df = pd.read_csv(DIR+file)
                if file_df.values.shape > (15,150): #just a check
                    file_df = file_df.fillna('0')
                    #literal_eval is needed to read the coordinates as type tuple
                    #and not as string.
                    file_df = file_df.applymap(literal_eval) 
                    coordinate_zero.append(file_df.values)
                    names_file[counter] = DIR+file
                    counter+=1
                    
    return coordinate_zero, names_file


def fill_previous_forward(coord_zero):
    """
    It fills the coordinate that are zero (null, not found keypoints)
    with the respective keypoint's coordinate in the previous frame.
    When no previous values are available, it fills in with the forward 
    values.
    
    Parameters
    ----------
    coord_zeros: list of arrays
        Each array is the coordinate matrix for a video where 
        the null keypoints are filled with zeros. 
    
    Returns
    --------
    keypoints_filled: list of arrays
        Each array is the coordinate matrix for a video where 
        the null keypoints are filled with previous/forward
        keypoint coordinate.
    
    """
    coord_zeros = coord_zero.copy()
    #Filling zeros with previous value
    keypoints_filled = []
    for n_video in range(len(coord_zeros)): 
        mask = coord_zeros[n_video]==0
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        np.maximum.accumulate(idx,axis=1, out=idx)
        coord_zeros[n_video] = coord_zeros[n_video][np.arange(idx.shape[0])[:,None], idx]
    #filling zeros with forward value (if the first one is nan...)
    for n_video in range(len(coord_zeros)): 
        mask = coord_zeros[n_video]==0
        idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
        idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
        keypoints_filled.append(coord_zeros[n_video][np.arange(idx.shape[0])[:,None], idx])
 
    return keypoints_filled


def not_swapped(keypoints_filled):
    """
    Swap the coordinate for the keypoints that have a right and left
    that are wrongfully assigned by OpenPose.
    
    Parameters
    -----------
    keypoints_filled: array
    
    Returns
    --------
    keypoint_not_swap: array
    """
    
    lst_videos=[]
    for video in keypoints_filled:

        dict_coor = {}
        for t in range(150):
            points = video[:, t]
            non_swap_points = []
            
            for i in range(16): 
                non_swap_points.append((0,0))
                
            #A = np.empty((15,), dtype=object)
            #add the head, that point never changes
            non_swap_points[0] = points[0]
            #add the neck, that point never changes
            non_swap_points[1] = points[1]
            #add the torso, that never changes
            non_swap_points[14] = points[14]
            #add the background, that never changes
            non_swap_points[15] = points[15]


            
            if points[2][0]<points[5][0]:
                non_swap_points[2] = points[2]
                non_swap_points[5] = points[5]
            else:
                #swap the shoulders
                non_swap_points[2] = points[5]
                non_swap_points[5] = points[2]
                    
            if points[3][0]<points[6][0]:
                non_swap_points[3] = points[3]
                non_swap_points[6] = points[6]
            else:
                #swap the elbows
                non_swap_points[6] = points[3]
                non_swap_points[3] = points[6]

            
            if points[4][0]<points[7][0]:
                non_swap_points[4] = points[4]
                non_swap_points[7] = points[7]
            else:
                #swap the wrist
                non_swap_points[7] = points[4]
                non_swap_points[4] = points[7]
               
            if points[8][0]<points[11][0]:
                non_swap_points[11] = points[11]
                non_swap_points[8] = points[8]
            else:
                #swap the hips
                non_swap_points[8] = points[11]
                non_swap_points[11] = points[8]

            if points[9][0]<points[12][0]:
                non_swap_points[9] = points[9]
                non_swap_points[12] = points[12]
            else:
                #swap the knees
                non_swap_points[12] = points[9]
                non_swap_points[9] = points[12]

            if points[10][0]<points[13][0]:
                non_swap_points[10] = points[10]
                non_swap_points[13] = points[13]
            else:
                #swap the ankle
                non_swap_points[13] = points[10]
                non_swap_points[10] = points[13]

            
            dict_coor[t] = non_swap_points
        
        lst_videos.append(np.array(pd.DataFrame(dict_coor)))
    
    array_videos = np.array(lst_videos)
        
    return array_videos

def smooth_coordinate(keypoints_coord,window_length=25,polyorder=2):
    """
    Smooth the keypoint coordinates using the savgol filter.
    
    Parameters
    ----------
    keypoints_matrix: array
    window_length, polyorder: int
        Parameter of the savgol filter: 
        length of the filter window (i.e., the number of coefficients),
        order of the polynomial used to fit the samples.
        
    Returns
    -------
    keypoints_matrix: array
    """
    keypoints_matrix = keypoints_coord.copy()
    for n_video in range(len(keypoints_matrix)):
        for i in range(15): 
            x_coord, y_coord = zip(*keypoints_matrix[n_video].T[:,i])
            x_coord = signal.savgol_filter(x_coord, window_length, polyorder)
            y_coord = signal.savgol_filter(y_coord, window_length, polyorder)
            x_coord = np.array(x_coord, int)
            y_coord = np.array(y_coord, int)
            keypoints_matrix[n_video].T[:,i] = list((zip(x_coord, y_coord)))
            
    return keypoints_matrix


def creating_X(keypoints_matrix):
    """
    From the keypoint matrix it retrives the 
    euclidean distance of the joints in time. The joint 
    pairs are collected in the 
    global variable DISTANCE.
    
    Parameters
    -----------
    keypoints_matrix: array
    
    Returns
    --------
    X: array
    """
    print("Building X")
    X = []
    for video in keypoints_matrix:
        dict_distance = {}
        for num_frame in range(150):
            dis = []
            for pair in DISTANCE: 
                partFrom = pair[0]
                partTo = pair[1]
                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]
                coord_1 = video[idFrom,num_frame]
                coord_2 = video[idTo,num_frame]
                if type(coord_1) == str:
                    coord_1 = eval(coord_1)
                if type(coord_2) == str:
                    coord_2 = eval(coord_2)
                dis.append(distance.euclidean(coord_1,coord_2))
            dict_distance[num_frame] = dis
        X.append(pd.DataFrame(dict_distance).values[:,:,np.newaxis])
        
    X = np.array(X)
    return X 

def creating_y(directory):
    """
    From the directory in which are stored the
    OpenPose estimation, it creates the dependent 
    variable of our model. The y is composed
    by 0-6 label depending on which squat category 
    the video belongs to.
    
    Parameters
    ----------
    directory: directory 
    
    Returns
    -------
    y: array
    """
    
    #Finding the length of each folder in the directory
    folder_count = {}
    for folder in sorted(os.listdir(directory)):

        DIR = directory+folder+'/'
        c=0
        for file in sorted(os.listdir(DIR)):
            if not file.startswith('distance'):
                c+=1
        folder_count[folder] = c
    
    #Assigning a numerical label 0-6 (=position in the dictionary folder_count)
    #to each folder. Creating an array y in which each file has a label.
    y = []
    for index, (folder,count_file) in enumerate(folder_count.items()):
        y += [index for el in range(count_file)]
        
    y = np.array(y)
    return y


def load_Xy(directory,**kwargs):
    """
    This function applies to the keypoint coordinates the transformation done 
    during training (filling null coordinates with previous/forward value,
    swapping when right and left are wrongfully assigned, smoothing the pose
    with savgol filter). It creates the X, the Euclidean distance matrix between
    those transformes coordinates, and the respective y array in which are stored
    the label of the video. 
    
    Parameters
    ----------
    directory: directory
        Here, are stored the OpenPose output (it is a supdirectory of folder named
        as the squat category).
    Returns
    -------
    X: array of shape (number of video, number of distance,number of frame,1)
    Y: array of shape (number of video,1)
    """
    
    coord_zeros, names_files = fillna_coord_zeros(directory)
    coord_filled = fill_previous_forward(coord_zeros)
    coord_swapped = not_swapped(coord_filled)
    coord_smooth = smooth_coordinate(coord_swapped,**kwargs)
    
    X = creating_X(coord_smooth)
    y = creating_y(directory)
    
    return X, y, names_files