# Readme

## Basic Information
 Project : Vehicle Segmentation and Trajectory Management 
 
 Place : Universal Village Program @ MIT, supervised by Dr. Fang and Dr. Zhou
 
 Time : July and August in 2019
 
 Reference Paper: [A Method to Extract Overall Trajectory Information from Frame Sequence of Fixed Background without Object Tracking](https://ieeexplore.ieee.org/abstract/document/8642123)
 
## What Different Files For
|-- Code \
　　|-- README.md \
　　|-- 组会记录.txt \
　　|-- track　　　　　　　　　　　　　　　　　　　　　# Hongkai Chen's Code \
　　|　　|-- inertial.m \
　　|　　|-- pro.m \
　　|　　|-- readme.txt \
　　|　　|-- track.cpp \
　　|-- Trajectory　　　　　　　　　　　　　　　　　　　# My Own Code　\
　　|　　|-- point_cloud.py                       
　　|　　|-- trajectory.py \
　　|　　|-- video2pic.py 
 
## Used Video Clipes
Full video : [Loop.North.Zhongshan-West-G-1-20141028075000-20141028075941-1716171](https://drive.google.com/drive/folders/1W9AdAk36azt9QE6cmIQXET2HP-NAiFtI)
 - 'Zhongshan-West-cap2_',  # Vertical Straight, 01:16-02:03
 - 'Zhongshan-West-cap3',  # Left Truning, 00:04-00:31
 - 'Zhongshan-West-cap4',  # Horizontal Straight, 00:34-01:14
 - 'Zhongshan-West',  # First 5000 Frames
 
 ## Used Method
 - Image Moments : [Wikipedia](https://en.wikipedia.org/wiki/Image_moment), [OpenCV](https://docs.opencv.org/4.1.0/dd/d49/tutorial_py_contour_features.html)
 - All Clustering Method : [Scikit-Learn](https://scikit-learn.org/stable/modules/clustering.html)
 - K-Means
 - DBSCAN : [Tutorial](https://towardsdatascience.com/dbscan-algorithm-complete-guide-and-application-with-python-scikit-learn-d690cbae4c5d)
