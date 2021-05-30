# -*- coding: utf-8 -*-

"""

This is the script to extract the keypoints using the OpenPose model compiled
through OpenCV and trained on MPI dataset.
More info below the class

"""
                
import time
import cv2 as cv
import os
import pandas as pd
import math
from scipy.spatial import distance


# standalone function called by the subsequent class
def distance_point(body_parts):
    """
    It associates through a unique id each body_part to
    each other body_part.
    
    Parameters
    ----------
    body_parts: dictionary in the form {KEYPOINT: unique_number,..}
    
    Returns
    --------
    body_distance: dictionary
        Returns all possible pair combinations of the keypoints. It is 
        dictionary of the form {(KEYPOINT_1,KEYPOINT_2): unique_number,..}.

    """
    
    body_parts = list(body_parts.keys())
    
    # the model also detects the 'background' as a keypoint, not of interest
    body_parts.remove("Background")

    counter = 1
    lst = []

    for part in body_parts:
        for index in range(counter,len(body_parts)):
            lst.append((part,body_parts[index]))

        counter +=1
    
    body_distance = {i: aux for aux, i in enumerate(lst)}
    
    assert(math.factorial(len(body_parts))/((2)*math.factorial(len(body_parts)-2)) ==len(body_distance))
    
    return body_distance

# Define Global variables
BODY_PARTS={ "Head":0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
    "Background": 15 }

DISTANCE = distance_point(BODY_PARTS)

#same as BODY_PARTS but key and values swapped
BODY_PARTS_swp = {value:key for key, value in BODY_PARTS.items()}

class PoseRetrivalAll:
    
    def __init__(self,
                 input_dir='../Video_Dataset/',
                 list_folder=['good'],
                 subfolder = '',
                 output_folder = 'pose_output',
                 output_dir = os.getcwd()+'/..',
                 protoFile='../model/pose_deploy_linevec_faster_4_stages.prototxt',
                 weightsFile='../model/pose_iter_160000.caffemodel'):
        """
        Parameters
        ----------
        input_dir: os.dir (default:'../Video_Dataset/' )
            Directory containing subfolders with the videos
            
        list_folder: list (default: ['good'])
            Name of the folders in input_dir containing the videos 
        
        subfolder: os.dir(default:'')
            Name of the eventual subfolder that contains the videos.
            Relevant if path to videos is like: input_directory/list_folder[i]/subfolder/video.mp4
            Set to '' if there aren't subfolders after folders in list_folder
        
        output_folder: string (default: 'pose_output')
            Name of the output folder in which is desired to store 
            the sub-folders named as the squat classification.
            
        output_dir: os.dir (default: 1 level below curr dir)
            Directory where to create folder with output 
            
        protoFile: os.dir (default:'../model/pose_deploy_linevec_faster_4_stages.prototxt')
            Directory of the model architecture
        
        weightsFile: os.dir (default:'../model/pose_iter_160000.caffemodel')
            Directory of the weights trained on MPI dataset
        """
        
        self.input_dir   = input_dir
        self.list_folder = list_folder
        self.output_folder = output_folder
        self.output_dir  = output_dir
        self.protoFile   = protoFile
        self.weightsFile = weightsFile
        self.subfolder   = subfolder
        
        # Load the network model and the corresponding pre-trained weights
        self.net = cv.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        
        print('Pose retrival instantiated')
    
    def info(self):
        """
            This class runs the pose estimator for each specified folder
            within the input main directory, looking for videos to process. 
            Returns 2 CSV file for each video: one containing the keypoint 
            coordinates for each frame, the other containing the distance from 
            each keypoint to each other, for every frame.
        """
        return
    
    def model(self,
              resized=True,
              resized_width=240,
              resized_height=400,
              divider = 2,
              thr=0.15,
              inSize=(368,368),
              scaling_factor=1.0/255,
              mean_channel=(104,116.67,122.68),
              distances = False):
        
        """
        For each video, it retrieves the coordinates of 15 keypoints (in time) 
        detected by OpenPose model.
        Those matrices are stored in separate CSV files saved in subfolders 
        within a folder called 'pose_output'. Subfolders correspond to the input 
        folder names. All output folders are automatically created.
        CSVs with keypoints coordinates are stored with the same name as the 
        input file, whereas CSV with keypoint distances are saved as 
        'distance_' + name of the input file + .csv.
        
        Notes
        --------
        Coordinates dataframe have dimensions of (15 * num_frame). 15 is the
        number of keypoints individued by the MPI dataset.
        
        Distance dataframe have dimensions of (105 * num_frame). 105 is the 
        number of unique pairs between the 15 keypoints.
        
        
        Parameters
        -----------
        resized: boolean (default:True)
            True means the input videos will be resized before being fed to the 
            algorithm
        
        resized_width: int (default: 240)
            If 'resized' is set True, it represents the width to which to resize
        
        resized_height: int (default: 400)
            If 'resized' is set True, it represents the height to which to 
            resize
            
        divider: int (default: 2)
            If num_frame%divider = 0, the frame will be skipped.
            Set to 2 to skip even frame, to 1 if no frame has to be skipped
        
        thr: float (default: 0.15)
            The openpose model output a confidence map on the keypoints'
            coordinates. We accept those coordinates that have confidence 
            higher that the threshold
        
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
       
        """
        
        print("Starting pose extraction")
        start_time = time.time()
        
        # create the main output folder, if not already there
        if self.output_folder not in os.listdir(self.output_dir):
            os.mkdir(self.output_dir + '/'+ self.output_folder)
        
        # identify folders in the main directory
        for folder in os.listdir(self.input_dir):
            
            # look only at the specified folders
            if folder in self.list_folder:
                
                # create the output folder for this iteration,
                # if such folder is not already present
                if folder not in os.listdir(self.output_dir+'/'+self.output_folder):
                    os.mkdir(self.output_dir+'/'+self.output_folder +'/'+folder)
                    
                
                # store the path to the folder containing videos
                DIR = self.input_dir+ '/' + folder + '/' + self.subfolder
                
                # clarify a common mistake
                try:
                    os.listdir(DIR)
                except OSError as err:
                    print("\nERROR double-check the subfolder argument passed to the model")
                    print("probably you want to set it to '', or remove the initial / \n")
                    print(err)
                    
                
                # start the keypoint extraction for each video in the folder
                for element in os.listdir(DIR):
                    
                    # read the video                     
                    cap = cv.VideoCapture(DIR+'/'+element)
                    
                    # initialize video-specific variables
                    num_frame = 0
                    dict_coordinate = {}
                    dict_distance = {}
                    
                    # a slightly more nuanced infinite loop
                    while cv.waitKey(1) < 0:
                        
                        # read the video, outputting an image (frame)
                        # and a bool confirming that frame is a valid image
                        hasFrame, frame = cap.read()
                        
                        # if no image can be correctly extracted, it means that
                        # the final frame has been reached, break the loop
                        if not hasFrame:
                            cv.waitKey()
                            break
                        
                        num_frame += 1
                        
                        # skip each-other frame
                        if num_frame % divider != 0:
                            continue 
            
                        else:
                            
                            # eventually resize the image 
                            if resized:
                                frame = cv.resize(frame,(resized_width,resized_height))
                            frameWidth = frame.shape[1]
                            frameHeight = frame.shape[0]
                            
                            # Feed the img to the newtwork while adjusting it with a scaling factor (1.0 = not scaling), 
                            # inSize is the size that the network excpects,
                            # then we have a tuple for mean subtraction (RGB respective means to subtract; this operation somewhat standardizes images)
                            # Finally swap the R and B channel to pass from RGB to BGR (input to the network must follow BGR convention)
                            self.net.setInput(cv.dnn.blobFromImage(frame,scaling_factor, 
                                                              inSize,mean_channel,
                                                              swapRB=True,crop=False))
                                                              
                            # get the keypoints coordinates, we care only about the first 16 keypoints
                            out = self.net.forward()
                            out = out[:, :16, :, :] 
                            
                            # subsequent loop stores the coordinates for each keypoint and calculates the distance matrix 
                            points = []
                            for i in range(len(BODY_PARTS)):
                                # Slice heatmap of corresponging body's part
                                heatMap = out[0, i, :, :]
                                # minMaxLoc returns the coordinates and the values of the min and max values in an array. 
                                # In this case we are only interested in max vals.
                                # ------------
                                # Since heatmap is a 46x46 array containing the probabilities of the keypoint being in
                                # a particular point of the grid (in a certain pixel), finding the max probability 
                                # allows us to identify the relative coordinates of the keypoint in a similar 
                                # way to how argmax works. The max value itself is the probability, so the confidence
                                # we have in saying that that pixel is indeed the keypoint of a certain bodypart
                                _, conf, _, point = cv.minMaxLoc(heatMap)
                                
                                # The heatmap matrix is a standard 46x46 grid; we now map back from
                                # the relative position in the grid to the actual location of the points in
                                # the image
                                x = (frameWidth * point[0]) / out.shape[3]
                                y = (frameHeight * point[1]) / out.shape[2]
                                # store only the points whose confidence score is higher than the threshold
                                points.append((int(x), int(y)) if conf > thr else None)
                                
                                # this assert is to prove that the confidence is just the value of the greatest point in the heatMap
                                # assert(  max(heatMap[point[1]]) == conf)
                                
                                # what we have just seen can, of course, be put into a oneliner, though it's not worth it!
                                # points = [( int( (frameWidth * cv.minMaxLoc(out[0, i, :, :])[3][0]) / out.shape[3]), int( (frameHeight * cv.minMaxLoc(out[0, i, :, :])[3][1]) / out.shape[2]) )  if  cv.minMaxLoc(out[0, i, :, :])[1] > thr else None  for i in range(len(BODY_PARTS))]
                            
                            dict_coordinate[num_frame] = points
                            
                            if distances: #Retriving distances --no optimal--
                                dis =[]
                                # form the pairings for the distance matrix and calculate
                                # the euclidian distances
                                for pair in DISTANCE: 
                                    partFrom = pair[0]
                                    partTo = pair[1]
                                    idFrom = BODY_PARTS[partFrom]
                                    idTo = BODY_PARTS[partTo]
                                
                                    if points[idFrom] and points[idTo]:
                                                
                                        dis.append(distance.euclidean(points[idFrom],points[idTo]))
                                        
                                    else:
                                        dis.append(None)
                                
                                # oneliner of the above (not worth it :s)        
                                # dis = [ distance.euclidean(points[self.BODY_PARTS[pair[0]]], points[self.BODY_PARTS[pair[1]]]) if points[idFrom] and points[idTo] else None for pair in pairings ]
                                
                                dict_distance[num_frame] = dis
                            
                        
                    # form the dfs and store them as CSV    
                    df_coordinate = pd.DataFrame(dict_coordinate)
                    file_name = os.path.splitext(element)[0] +'.csv'
                    df_coordinate.to_csv(self.output_dir +'/'+ self.output_folder+'/'+folder+'/'+file_name, index=False)
                    
                    if distances:
                        df_distance = pd.DataFrame(dict_distance)
                        df_distance.to_csv(self.output_dir +'/'+ self.output_folder+'/'+folder+'/'+'distance_'+file_name, index=False)
                    

        
        print(f"ALL poses retrieved in: {round((time.time() - start_time)/3600,2)} hours")
    
    
    
    def show_video(self, video_path, thr=0.15):
        """
        Shows the video with the plotted keypoints
                
          Parameters
          -----------
          video_path: str 
              path to the video to be visualized with keypoints
          
          thr: float (deafault: 0.15)
              only keypoints whose prediction confidence is above this
              threshold will be shown
        """
        
        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], 
                       ["RShoulder", "RElbow"], ["RElbow", "RWrist"], 
                       ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                       ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], 
                       ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
        
        inWidth = 368
        inHeight = 368

        cap = cv.VideoCapture(video_path)
        
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv.waitKey()
                break
                
            frame = cv.resize(frame, (240*2,800))
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            
            self.net.setInput(cv.dnn.blobFromImage(frame, 
                                                   1.0 /255, 
                                                   (inWidth, inHeight), 
                                                   (0, 0, 0), 
                                                   swapRB=False, 
                                                   crop=False))
            out = self.net.forward()
            out = out[:, :len(BODY_PARTS), :, :]  
            
            points = []
            for i in range(len(BODY_PARTS)):
                heatMap = out[0, i, :, :]
                _, conf, _, point = cv.minMaxLoc(heatMap)
        
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                points.append((int(x), int(y)) if conf > thr else None)   
                
            # points = [( int( (frameWidth * cv.minMaxLoc(out[0, i, :, :])[3][0]) / out.shape[3]), int( (frameHeight * cv.minMaxLoc(out[0, i, :, :])[3][1]) / out.shape[2]) )  if  cv.minMaxLoc(out[0, i, :, :])[1] > thr else None  for i in range(len(BODY_PARTS))]

            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]    
        
                if points[idFrom] and points[idTo]:
                    cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360,
                               (0, 0, 255), cv.FILLED)
                    cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360,
                               (0, 0, 255), cv.FILLED)
        
            t, _ = self.net.getPerfProfile()
            cv.imshow('',frame)