# Emotion-Recognition
This python module implements Emotion Recognition with Dense Optical Flow, K Nearest neighbours and Hidden Markov Models.

First dense optical flow will be calculated and using PCA dimensionality of optical flow is reduced. Then two stage classification is made on reduced optical flow. In first stage K Nearest neighbours is used to classify optical flow of each image. In second stage Hidden markov model is used to classify video by making use of classifications of images in first stage classification.

This is implemeted using "Recognition of Facial Expressions and Measurement of Levels of Interest From Video" by
Mohammed Yeasin, Baptiste Bullot and Rajeev Sharma, (2006). [Link to paper, IEEE](http://ieeexplore.ieee.org/abstract/document/1632035/).
