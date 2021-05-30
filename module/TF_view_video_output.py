# -*- coding: utf-8 -*-
"""
The code below serves the purpose of visualizing the pose-extractor predictions
together with the videos, on Colab

@author: Eugen
"""
def plot_output(video_paths, cancel_background=False):
    """ 
    Loop through all the video frames.
    Extract at each step, (x,y) coordinates for all keypoints 
    for which confidence > threshold.
    Plot them in an image corresponding to the frame
    
    Parameters
    ----------
    video_paths: directory
    cancel_background: bool
      True if one wants to visualize the predictions on a black screen
    
    Returns
    ----------
    plots images of the frames together with keypoints  predictions
    """


# loop through all videos received as input
    for video_path in video_paths:

# setup the estimator 
      w, h = model_wh('432x368')
      e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))

# open the video
      print("\nopening the following video: ", video_path, "\n")
      cap = cv2.VideoCapture(video_path)
      if cap.isOpened() == False:
          print("Error opening video stream or file")

# setup some variables
      number_of_the_frame = 0  

# loop through the frames of the video
      while cap.isOpened():
          ret_val, image = cap.read()
          number_of_the_frame += 1

  # break the loop if we aren't getting any more video frames
          if not ret_val:
            cv2.waitKey()
            break

  # start a try-except flow just in case
          try:

  # extract the object containing all relevant info
            humans = e.inference(image,upsample_size=4.0)

  # unsilence if you want to plot keypoints in a not noisy environment (obscure everything)
          if  cancel_background:
              image = np.zeros(image.shape)

  # extract the image for plotting purposes
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

  # catch exceptions
          except Exception as err:
            print(f"\nthere was an error at frame {number_of_the_frame}, skipping frame because:")
            print(err)
            pass        

 # print the image for this frame
          cv2_imshow(image)
          cv2.destroyAllWindows()
          
    return 