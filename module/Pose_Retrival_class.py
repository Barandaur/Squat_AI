# -*- coding: utf-8 -*-
"""

This file run the pose estimator for just a single input file.

"""

import time
import cv2 as cv
import os
import pandas as pd
import math
from scipy.spatial import distance
from Pose_Retrival_All_class import distance_point

BODY_PARTS = {"Head":0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
            "Background": 15 }
            
DISTANCE = distance_point(BODY_PARTS)

# Detailed expalanation of this code in the pose_retrival_all_class module
class PoseRetrival:
    def __init__(self, 
                 input_video,
                 protoFile='../model/pose_deploy_linevec_faster_4_stages.prototxt',
                 weightsFile='../model/pose_iter_160000.caffemodel'):
      """
        Parameters
        ----------
        input_video: str 
            Path of the file
    
        protoFile: os.dir (default:'../model/pose_deploy_linevec_faster_4_stages.prototxt')
            Directory of the model architecture
        
        weightsFile: os.dir (default:'../model/pose_iter_160000.caffemodel')
            Directory of the weights trained on MPI dataset
        """
        
        self.input_video = input_video
        self.protoFile = protoFile
        self.weightsFile = weightsFile
        print('Pose retrival instantiated')
    
    def model(self,
              resized=True,
              resized_width=240,
              resized_height=400,
              thr=0.15,
              inSize=(368,368),
              scaling_factor=1.0/255,
              mean_channel=(104,116.67,122.68),
              distances = False,
              all_frame=False, 
              save_df = True):
        
        """
        For each video, it retrieves the coordinates of 16 keypoints (in time) 
        detected by OpenPose model. Those matrices are stored in separate CSV 
        files saved in subfolders within a folder called 'pose_output'.
        Subfolders correspond to the input folder names. 
        All output folders are automatically created.
        CSVs with keypoints coordinates are stored with the same name as the 
        input file.
        
        Notes
        ----------
        Coordinates dataframe have dimensions of (16 * num_frame). 16 is the
        number of keypoints individued by the MPI dataset.
        
        
        Parameters
        ----------
        resized: boolean (default:True)
            True means the input videos will be resized before being fed to the 
            algorithm
        
        resized_width: int (default: 240)
            If 'resized' is set True, it represents the width to which to resize
        
        resized_height: int (default: 400)
            If 'resized' is set True, it represents the height to which 
            to resize
        
        thr: float (default: 0.15)
            The openpose model output a confidence map on the keypoints' 
            coordinates. We accept those coordinates that have confidence higher 
            that the threshold
        
        inSize: tuple (default: (368,368))
            Input size needed for the model architecture
        
        scaling_factor: float (default:1.0/255)
            Inverse of the scaling factor
        
        mean_channel: tuple (default:(104,116.67,122.68))
            Mean scalar to be subtracted from the color channels in order to
            normalize the image. It is written in (R, G, B) channels.
            
        distances: bool (default: False)
            If True it retrieves euclidian distances among the keypoints
            together with the coordinates. It will save the output in a
            csv file starting with 'distance_' + name of the input file
        
        all_frame: bool (default: False) 
            If True it retrives all frame from the video, if False it will 
            retrive half of the frame (usually 15 frame per second)
              
        save_df: bool (dafault: True)
            If True it will save the coordinates as a csv file.

        Return
        ----------
        df_coordinates: array
              Return as output a matrix of size 16 x num_frame that 
              stores all the coordinates.
        """  
        start_time = time.time()
        
        net = cv.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        cap = cv.VideoCapture(self.input_video)
        print('loading')

        num_frame = 0
        dict_coordinate = {}
        dict_distance = {}

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv.waitKey()
                break
            
            num_frame += 1
            if num_frame %2 == 0 and all_frame == False:
                continue 
            else:
                if resized:
                    frame = cv.resize(frame,(resized_width,resized_height))
                frameWidth = frame.shape[1]
                frameHeight = frame.shape[0]
                net.setInput(cv.dnn.blobFromImage(frame, scaling_factor, 
                                                  inSize, 
                                                  mean_channel, swapRB=True, 
                                                  crop=False))
                out = net.forward()
                out = out[:, :16, :, :] 
                points = []
                for i in range(len(BODY_PARTS)):
                    heatMap = out[0, i, :, :]
                    _, conf, _, point = cv.minMaxLoc(heatMap)
            
                    x = (frameWidth * point[0]) / out.shape[3]
                    y = (frameHeight * point[1]) / out.shape[2]
                    points.append((int(x), int(y)) if conf > thr else None)
                
                dict_coordinate[num_frame] = points
        
        print(f"Pose retrieved in: {round((time.time() - start_time)/60,2)} minutes")
        df_coordinate = pd.DataFrame(dict_coordinate)

        if save_df:
            file_name = os.path.basename(self.input_video)
            file_name = os.path.splitext(file_name)[0] +'.csv'
            df_coordinate.to_csv(file_name,index=False)        
            
        return df_coordinate.values