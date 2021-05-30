# -*- coding: utf-8 -*-

from ast import literal_eval
import os 
import pandas as pd
import numpy as np
from scipy import signal
from scipy.spatial import distance
from Pose_Retrival_class import *
import cv2 as cv

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

# Utilis to pre-process the data
def fill_previous_forward(coord_zeros):
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
    coord_zeros = np.nan_to_num(coord_zeros)
    #Filling zeros with previous value
    mask = coord_zeros==0
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    coord_zeros = coord_zeros[np.arange(idx.shape[0])[:,None], idx]
    
    #filling zeros with forward value (if the first one is nan...)
    mask = coord_zeros==0
    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    coord_zeros = coord_zeros[np.arange(idx.shape[0])[:,None], idx]
 
    return coord_zeros

def not_swapped(video):
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
    
    array_video = np.array(pd.DataFrame(dict_coor).values)
        
    return array_video

def smooth_coordinate(keypoints_matrix,window_length=25,polyorder=2):
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
    for i in range(15): 
        x_coord, y_coord = zip(*keypoints_matrix.T[:,i])
        x_coord = signal.savgol_filter(x_coord, window_length, polyorder)
        y_coord = signal.savgol_filter(y_coord, window_length, polyorder)
        x_coord = np.array(x_coord, int)
        y_coord = np.array(y_coord, int)
        keypoints_matrix.T[:,i] = list((zip(x_coord, y_coord)))
        
    return keypoints_matrix


def creating_X(video):
    """
    From the keypoint matrix it retrives the euclidean distance of the joints in time. 
    The joint pairs are collected in the global variable DISTANCE.
    
    Parameters
    -----------
    video: array
    
    Returns
    --------
    X: array
    """
    #print("Building X")
    X = []

    dict_distance = {}
    for num_frame in range(150):
        dis = []
        for pair in DISTANCE: 
            partFrom = pair[0]
            partTo = pair[1]
            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
            coord_1 = video[idFrom, num_frame]
            coord_2 = video[idTo, num_frame]
            if type(coord_1) == str:
                coord_1 = eval(coord_1)
            if type(coord_2) == str:
                coord_2 = eval(coord_2)
            dis.append(distance.euclidean(coord_1,coord_2))
        dict_distance[num_frame] = dis
        
    X = np.array(pd.DataFrame(dict_distance).values).reshape(1, 105, 150, 1)
    return X 
    
import time

def keypoints_visualiation(video_input, data_input):
    """
    Visualize the video point 
    """
    df_coordinate = pd.DataFrame(data_input)
    cap = cv.VideoCapture(video_input)
    s=0
    t=0
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        time.sleep(0.01)
        if not hasFrame:
            cv.destroyAllWindows()
            break
        frame = cv.resize(frame, (240,400))
        s+=1

        if s%2!=0 and t<150:
            for pair in POSE_PAIRS:
                partA = BODY_PARTS[pair[0]]
                partB = BODY_PARTS[pair[1]]
                cv.line(frame, df_coordinate.loc[partA, t], df_coordinate.loc[partB, t], (255, 0, 0), 3)
                cv.ellipse(frame, df_coordinate.loc[partA, t], (3, 3), 0, 0, 360, (255, 0, 255), cv.FILLED)
                cv.ellipse(frame, df_coordinate.loc[partB, t], (3, 3), 0, 0, 360, (255, 0, 255), cv.FILLED)
            t+=1
            cv.imshow('',frame)

def load_X(video, df = False, visualize = False):
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
    if df:
        file_df = pd.read_csv(video)
        file_df = file_df.fillna('0')
        file_df = file_df.applymap(literal_eval) 
        data = file_df.values
    else:
        retrival = PoseRetrival(video)
        data = retrival.model(save_df = False)
    #Filling
    data = fill_previous_forward(data)
    #Swapping
    data = not_swapped(data)
    #Smoothing
    data = smooth_coordinate(data)
    #Calculating Distances
    data_dist = creating_X(data)

    if visualize:
        visual = video[:-3]+'mp4'
        keypoints_visualiation(visual, data)
    
    return data_dist


# Utils for prediction
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger


def relu_bn(inputs) :
    """
    After each conv layer, a Relu activation and a Batch Normalization are 
    applied

    Parameters
    ----------
    inputs: Tensor  Input tensor from the previous layer

    Returns
    --------
    bn: batch normalized layer

    """
    
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)

    return bn

def residual_block(x, downsample, filters, kernel_size = 3):
    """
    This function constructs a residual block. It takes a tensor x as input and 
    passes it through 2 conv layers

    Parameters
    ----------
    x: Tensor Input tensor from the previous layer (or Input if it is the
    first one)

    downsample: bool When true downsampling is appplied: the stride of the 
    first Conv layer will be set to 2 and the kernel size passes from 3 to 1

    filters: int Number of filters applied to the data

    kernel_size: int Kernel size, default value equals to 3
    
    Return
    --------
    out: layer of the residual block calculated
    """

    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)

    y = relu_bn(y)

    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    # With the downsaple parameter set to True:
    # the strides of the first Conv are set to 2
    # the kernel size of the conv layer on the input passes from 3 to 1
    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)

    # The input x is added to the output y, and then the relu activation and 
    # batch normalization are applied           
    out = Add()([x, y])
    out = relu_bn(out)

    return out

def create_res_net(optimizer):
    """
    A function to create the ResNet. It puts together the two other functions 
    already defined
    
    Parameters
    ----------
    optimizer: str The name of the optimizer we want to use to compile the model

    Return
    ----------
    Model initalized
    """

    # The dimension of our distance matrices
    inputs = Input(shape=(105, 150, 1))

    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    # The residual function is called to add the skip connections
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t,
                               downsample=(j==0 and i!=0), 
                               filters=num_filters)

        # The number of filters applied to the residual blocks are increasing 
        # (64,128,256,512)    
        num_filters *= 2
    
    # Average pooling layer to reduce the dimension of the input by computing 
    # the average values of each region
    t = AveragePooling2D(4)(t)

    # Flatten layer 
    t = Flatten()(t)

    # Dense layer to produce the probabilities of all the classes 
    outputs = Dense(7, activation='softmax')(t)

    # Initalizing the model
    model = Model(inputs, outputs)

    # Sparse categorical crossentropy since we do not have one-hot arrays for 
    # the probabilities
    model.compile(optimizer= optimizer,
                  loss= 'sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model