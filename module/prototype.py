# -*- coding: utf-8 -*-
from utils_prediction import load_X, create_res_net
import numpy as np
import argparse

parser = argparse.ArgumentParser('''Give me a squat and I will tell you if it is correct''')
parser.add_argument('--v', help='Video you want to classify, it should be of at least 10s!', type=str)
parser.add_argument('--p', help="Visualize the retrived key points in the video, default set to False", default=False, type=bool)
args = parser.parse_args()

video = args.v
visualize = args.p

#Transforming the Video
if video.endswith('csv'):
    data = load_X(video, df = True, visualize = visualize) 
else:
    data = load_X(video, visualize = visualize)


#Prediction
model = create_res_net('SGD')
model.load_weights('../weights/sgd32_weights.26-0.61.hdf5')
prediction = model.predict(data)
prediction = np.argmax(prediction, axis = 1)

if prediction == 6:
    print('You have done an excellent squat!')
elif prediction == 5:
    print('Watch out your warp when you are going down!')
elif prediction == 4:
    print('You should go more down, this is not a good squat!')
elif prediction == 3:
    print("Don't move your thigh when you are going down!")
elif prediction == 2:
    print('Eyes up front, young man!')
elif prediction == 1:
    print("Keep you feet on the floor if you want to do a good squat!")
else:
    print('Watch out your back!')

