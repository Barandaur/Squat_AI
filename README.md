# Squat.AI

## Our Team
This work couldn't have been possible without an amazing team, so acknowledgements go to: 
- Alessandro Pisano ([Linkedin](https://www.linkedin.com/in/alessandro-pisano-276048161/), [GitHub](https://github.com/alessandro-pisano)), 
- Caterina Fabbri ([Linkedin](https://www.linkedin.com/in/caterina-fabbri/), [GitHub](https://github.com/CaterinaFabbri)), 
- Chiara di Bonaventura ([Linkedin](https://www.linkedin.com/in/chiara-di-bonaventura/), [GitHub]()), 
- Flavia Monaci ([Linkedin](https://www.linkedin.com/in/flavia-monaci-76503319a/), [GitHub]())

## Project Description
As coursework for the Computer Vision exam, my team produced a script which takes videos of a squatting person as input, and returns whether the exercise is correctly done or, otherwise, feedback on the mistake.

We used a similar approach to Ogata et al. (2019). Notably, our main strategy was to carry out a multi-class classification task based on temporal distance matrices computed on the 2D body joints retrieved from the videos.

## Pipeline
We take advantage of the publicly available dataset used in Ogata et al. (2019).
Our steps can be defines as follow:
- Every other frame of an input video, *extract the keypoint coordinates of the body*. To do so, we test two OpenPose implementation and evaluate them based on several factors, the main one being the number of missing values or frames that each model outputs.
- *Pre-processing*. We deal with several problems of the data extracted in the previous step. For example, we need to deal with NaNs (we ended up imputing them), with videos which may be registered at different fps, or with the coordinates of some body-parts getting mistakenly swapped. Finally, we smooth the keypoints time-series using the Savitzky-Golay Filter.
- *Create the input data*. Firstly, we compute the distance from each body-part to each other, at every extracted frame. Then, we create a matrix where these distances -in a given frame- are the columns, and each row is a frame (thus each input video is summarized into a matrix). As in Ogata et al. (2019), we use distances as input because they are a general representation (the information doesn't depend on lighting, background etc.).
- *Normalization*. Despite the previous step, subject-specific information still remains, such as limb-length, which varies depending on the individual. We address it evaluating several types of normalization (e.g. normalization by the lenght of the torso, as in Chen and Yang (2020) or by arm's length).
- *Model*. For our multi-class classification task, we  tested many algorithms: ResNet, AlexNet, VGGNet and LSTM-CNN models. We chose these models because they are among the most common ones in the sector. ResNet ends up having the best performance. As a baseline, we also run an LSTM-CNN on the raw pixel-data from the videos.
- *Arg-parse application*. Finally, we make the code callable by a terminal window, and we test the model on independent videos made by us, to further evaluate how it generalizes.
