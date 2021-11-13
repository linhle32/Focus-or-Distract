###
#This module consists of utility functions for processing videos including
#  1. detect face region in each frame
#  2. crop all frames to a square region around the face
#  3. extract eye coordination and blinking states from a video

import dlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skvideo.io
import skimage.measure
import os
from gaze_tracking.gaze_tracking import GazeTracking

# function to convert dlib shapes to numpy arrays
def shape_to_np(shape):
    points = []
    for p in shape.parts():
        points.append([p.x, p.y])
    return np.array(points)

# function to detect the facial region in a single frame
# return the facial center and maximum distance from center to a boundary
def get_face_region(frame, detector, predictor, pad):
    #convert the frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get the facial region, assuming there is a single person in the video
    rect = detector(gray, 0)[0]
    shape = shape_to_np(predictor(gray, rect))
    # Get the center of the facial region        
    center = np.mean(shape[[0,27,16],:], axis=0, dtype=np.int32)
    # Get max distance from facial center to a boundary
    min_w, max_w, min_h, max_h = [shape[:,0].min(), shape[:,0].max(), shape[:,1].min(), shape[:,1].max()]
    max_dis = np.max([
                        min_w - center[0],
                        max_w - center[0],
                        min_h - center[1],
                        max_h - center[1],
                     ])
    max_dis += pad
    return center, max_dis 

# function to crop a frame to square
# use center coordination and max distance from get_face_region
def squarize_frame(frame, center, max_dis, new_size):
    #convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_frame = np.zeros([max_dis*2,max_dis*2])
    #region in original frame
    ori_min_x = max(0, center[0]-max_dis)
    ori_max_x = min(W, center[0]+max_dis)
    ori_min_y = max(0, center[1]-max_dis)
    ori_max_y = min(H, center[1]+max_dis)
    #region in new frame
    new_min_x = max(0, M - C[0])
    new_max_x = new_min_x + ori_max_x - ori_min_x
    new_min_y = max(0, M - C[1])
    new_max_y = new_min_y + ori_max_y - ori_min_y    
    #copy face to new frame
    new_frame[new_min_y:new_max_y, new_min_x:new_max_x] = gray[ori_min_y:ori_max_y, ori_min_x:ori_max_x]
    return cv2.resize(new_frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)


#function to extract eye coordinations and blinking statuses from a video
def extract_eye_info(video):
    #list to store eye coordinations and statuses
    eye_info = []
    for i in range(video.shape[0]):
        frame = video[i]
        #reshape back to three channels
        cf = np.repeat(frame,repeats=3).reshape(160,160,3)
        #extract eye data
        try:
            gaze.refresh(cf)
            le, re, lb, rb = gaze.pupil_left_coords(), gaze.pupil_right_coords(),gaze.eye_left.blinking, gaze.eye_right.blinking
            count = 0 #extraction may randomly fail, therefore retry up to 10 times
            while (count < 10 and (le is None and re is None)):
                gaze.reset()
                gaze.refresh(cf)
                le, re, lb, rb = gaze.pupil_left_coords(), gaze.pupil_right_coords(),gaze.eye_left.blinking, gaze.eye_right.blinking
                count += 1
            if (le is None and re is None): #if not detect anything after 10 tries, use data from previous frame
                miss += 1
                le = pre_le
                re = pre_re
            pre_le = le
            pre_re = re
        except: #if failed to detect at all, put 0 for coordinates and 9 for statuses
            le, re, lb, rb = (0, 0), (0, 0), 9, 9
        eye_info.append(np.hstack((le, re, lb, rb)))
    return np.array(eye_info)