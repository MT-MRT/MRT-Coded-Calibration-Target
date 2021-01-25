# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:06:18 2021
@author: ebert, schramm
"""

# import self-written functions
import featureFunctions as ff

# import standard libaries
import cv2
from cv2 import aruco
import numpy as np
import os
import yaml
import random

if __name__ == '__main__':

    # Load settings parameters
    with open('parameters.yaml') as file:
        parameter = yaml.safe_load(file)

    # Initialize empty camera parameter
    camera_matrix_mrt = np.array([],np.float32)
    dist_coeffs_mrt = np.array([0.0,0.0,0.0,0.0,0.0],np.float32)
    rmse = 0.0

    # Set AruCo parameter
    aruco_dict = ff.getDictonary(parameter)
    gridSize, arucoSize = parameter['gridSize'], parameter['arucoSize']

    # Load images and check for target
    img_list = sorted(os.listdir(parameter['path']))
    imgCorners = []
    img_all = []
    for img_name in img_list:
        img_all.append(cv2.imread(parameter['path'] + img_name))
        img_all[-1] = cv2.cvtColor(img_all[-1], cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_all[-1], cv2.COLOR_RGB2GRAY)
        corners, _, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco.DetectorParameters_create())
        if corners != []:
            imgCorners.append(np.swapaxes(np.asarray(corners[0][:,:,:]),0,1))
        else:
            print('No corner found in', img_name)
            del(img_all[-1])
    
    print('-' * 60)
    print('Target found in', len(img_all), 'images')

    # Start of statistical analysis
    rmse_all = []
    mtx_all = []
    dist_all = []
    
    # Loop over every run
    for i in range(parameter['runs']):
        print('-' * 60)
        print(i, 'from', parameter['runs'])

        # Generate a subsample of images for the calibration run
        num_sub = random.sample(range(len(img_all)),parameter['numberOfImages'])

        # Get the img- and objCorners for this run
        img_sub = [img_all[i] for i in num_sub]
        imgCorners_sub = [imgCorners[i] for i in num_sub]
        objCorners = [np.array([np.array([0.,0.,0.]),np.array([0.,arucoSize,0.]),np.array([arucoSize,arucoSize,0.]),np.array([arucoSize,0.,0.])]).astype(np.float32)]*len(imgCorners_sub)
        
        # Calculate initial camera parameters based on the 4 AruCo corners
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objCorners, imgCorners_sub, img_sub[0].shape[0:2], camera_matrix_mrt, dist_coeffs_mrt, flags=cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_ZERO_TANGENT_DIST)
        
        # Print the result
        ff.printResults(mtx, dist, ret)
        
        # Intitialize the oldPoints list
        oaop = []
        oaip = []

        # Run the calibration jobs defined in the parameters file
        for calibration in parameter['calibrations']:
            rmse, mtx, dist, oaop, oaip = ff.calibrateCoded(parameter, img_sub, calibration['rows'], calibration['columns'], mtx, dist, search_radius = calibration['sr'], plot = calibration['plot'], oldAllObjPoints = oaop, oldAllImgPoints = oaip)
        
        # Append the results to the lists
        rmse_all.append(rmse)
        mtx_all.append(mtx)
        dist_all.append(dist)

    # Calculate and print the means and stds of the statistical multi-calibration
    rmse_all = np.asarray(rmse_all)
    mtx_all = np.asarray(mtx_all)
    dist_all = np.asarray(dist_all)
    print('rmse Mean:', np.mean(rmse_all, axis = 0))
    print('rmse Std:', np.std(rmse_all, axis = 0))
    print('mtx Mean:', np.mean(mtx_all, axis = 0))
    print('mtx Std:', np.std(mtx_all, axis = 0))
    print('dist Mean:', np.mean(dist_all, axis = 0))
    print('dist Std:', np.std(dist_all, axis = 0))