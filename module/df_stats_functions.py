# -*- coding: utf-8 -*-
"""
This code serves to compare the results of the 
two PoseEstimator models that we tested
"""

import matplotlib.pyplot as plt
import numpy as np


def get_dicts():
    """
    Returns a mapping from body-parts to unique values and viceversa

    Returns
    --------
    body_parts: dict
        dict mapping each body-part to a unique value
    body_parts_swp: dict
        dict mapping a key to each body-part
    """
    
    body_parts = {"Nose":0, 
              "Neck": 1,

              "RShoulder": 2, 
              "RElbow": 3, 
              "RWrist": 4,

              "LShoulder": 5, 
              "LElbow": 6, 
              "LWrist": 7, 

              "RHip": 8, 
              "RKnee": 9,
              "RAnkle": 10, 

              "LHip": 11, 
              "LKnee": 12, 
              "LAnkle": 13,

              "Chest" : 14}

    # get also the swapped version of this dict
    body_parts_swp = {value:key for key, value in body_parts.items()}
    
    return body_parts, body_parts_swp




def adjust_df(df, body_parts_swp):
        """
        Parameters
        ----------
        df: pd.DataFrame 
            df with the keypoints as columns
            
        body_parts_swp: dict {key:body_part}
            dict associating a unique key to each body_part
        Returns
        --------
        a df without irrelevant keypoints and with 
        columns named after each body-part
        """
    
        # drop irrelevant keypoints if they are in the df
        irr_keyp = list({15,16,17}.intersection(set(df.columns)))
        df.drop(irr_keyp, inplace = True, axis=1)
    
        # transform index into body-part names
        df.rename(columns=body_parts_swp, inplace=True)
        return df


def miss_frames(df, num_frames = 150):
        """
        Calculates the % of missing frames.
        Assumes that the df has a column named 'filename', 
        containing a different value for each video
        
        Parameters
        ----------
        df: pd.DataFrame 
            df with the keypoints
            
        num_frames: int (default: 150)
            number of frames considered in each video
        
        Returns
        --------
        prints the percentage of missing frames, and outputs
        the number of missing and total frames 
        """
        
        num_videos = df.filename.nunique()
        # sum the num of retrieved frames for each video and
        # subtract the total num frames which should have been retrieved
        miss_frames = abs((df.fillna(1).groupby('filename').count().sum() - num_frames*num_videos).iloc[0])
        tot_frames = num_frames*num_videos
        print(f"We are missing {miss_frames} frames on {tot_frames}, {round(miss_frames*100/tot_frames,2)}%")
        return miss_frames, tot_frames
    
    
def miss_keypoints(df):
        """
        Parameters
        ----------
        df: pd.DataFrame 
            df with the keypoints
            
        Returns
        --------
        a df conveniently formatted to show the number   
        of missing keypoints for each body_part
        """
    
        miss_keypoints=df.isnull().sum(axis=1).sum()
        tot_keypoints= len(df)*len(df.columns)
        print(f"in total, we are missing {miss_keypoints} keypoints on {tot_keypoints}, {round(miss_keypoints*100/tot_keypoints,2)}%\n")
        print("missing keypoints by bodypart:\n")
        return df.isnull().groupby('filename').sum().reset_index(drop=True).rename(index={0:"Missing Vals"})
    
def plot_keyp_y_variation(df, body_parts_swp, video_names=[]):
        """
        Parameters
        ----------
        df: pd.DataFrame 
            df with the keypoints, relative to 3 or more videos
            
        body_parts_swp: dict {key:body_part}
            dict associating a unique key to each body_part
            
        video_names: list
            if specified, the plot will be relative to videos
            whose name is both in the list and in the df
            
        Returns
        --------
        plots the time-series of the y coordinate for each keypoint for each of
        3 videos.
        Missing values at time t are not considered, and the next non NaN
        value is assumed to be relative to time t.
        Thus, time-series with missing vals are trimmed, and will be shorter
        """
        
        # keep only wanted videos in the df, if so specified
        if video_names:
          df = df[df.filename.isin(video_names)]
          
        # manage the plotting  
        fig, m_axs = plt.subplots(1, 3, figsize=(25, 8))
        for c_ax, (c_name, n_rows) in zip(m_axs, df.groupby('filename')):
            for i in range(14):
                c_rows = list(filter(lambda v: v==v,  n_rows[body_parts_swp[i]].values))
                # if the df comes from csv we need this eval, otherwise we don't
                try:
                    c_rows = [eval(i) for i in c_rows]
                except:
                    pass
                y_cor = [y for x,y in c_rows]
                c_ax.plot(np.arange(len(y_cor)), y_cor, label='{}'.format(body_parts_swp[i]))
            c_ax.legend()
            c_ax.set_title(c_name)
            c_ax.set_xlabel("time")
            c_ax.set_ylabel("y coordinate")
    
    
def plot_keyp_var(df, body_parts_swp, body_parts, video_names=[]):
        """
        Parameters
        ----------
        df: pd.DataFrame 
            df with the keypoints, relative to 3 or more videos
            
        body_parts_swp: dict {key:body_part}
            dict associating a unique key to each body_part
        
        body_parts: dict {body_part:key}
            dict associating a unique body_part to a unique number
            
        video_names: list
            if specified, the plot will be relative to videos
            whose name is both in the list and in the df
            
        Returns
        --------
        plots the variation in terms of (x,y) coordinate of all the body-parts
        throughout the video, for up to 3 videos.
        """
        
        # keep only wanted videos in the df, if so specified 
        if video_names:
          df = df[df.filename.isin(video_names)]
        
        # manage the plotting
        fig, m_axs = plt.subplots(1,3, figsize=(25, 8))
        for c_ax, (c_name, n_rows) in zip(m_axs, df.groupby('filename')):
            for i in range(len(df.columns)-1):
                c_rows = list(filter(lambda v: v==v,  n_rows[body_parts_swp[i]].values))
                # if the df comes from csv we need this eval, otherwise we don't
                try:
                    c_rows = [eval(i) for i in c_rows]
                except:
                    pass
                if c_rows != [None] * len(c_rows):
                    x_cor = [x for x,y in c_rows]
                    y_cor = [y for x,y in c_rows]
                c_ax.plot(x_cor, y_cor, label='{}'.format(body_parts_swp[i]))
            c_ax.legend()
            c_ax.set_title(c_name)
            c_ax.set_xlabel("x coordinate values")
            c_ax.set_ylabel("y coordinate values")