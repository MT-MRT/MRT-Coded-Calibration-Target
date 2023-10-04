import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

# Print the results in the console
def printResults(mtx, dist, ret, gridHeight = -1, gridWidth = -1, sr = -1):
    print("Calibration executed\n")
    if sr != -1:
        print('Search radius:\n', sr, "\n")
    if gridHeight != -1 and gridWidth != -1:
        print('Target size:\n', gridHeight, gridWidth, "\n")
    print("Camera matrix:\n"+str(mtx)+"\n")
    print("distorsion coefs:\n"+str(dist)+"\n")
    print('RMSE:', ret)
    print("-"*60)

# Get the correct AruCo-dictionary
def getDetector(parameter):
    if parameter['aruco_dict'] == 'aruco.DICT_4X4_1000':
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    elif parameter['aruco_dict'] == 'aruco.DICT_3X3_1':
        aruco_dict = cv2.aruco.extendDictionary(1, 3)
    else:
        print('ERROR! The aruco dictionary is not defined!')
        exit()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    return detector

# Plot the obj- and imgPoints with pyplot
def plotFunc(img_num, targetObjPoints, projImgPoints, targetImgPointsSP, frame_markers):
    fig, (ax1, ax2) = plt.subplots(2,figsize=(7,10))
    title = str("[Nr. "+str(img_num)+"] ")
    fig.canvas.set_window_title(title)

    ax1.set_title("Object Points")
    ax1.set(xlabel='[mm]', ylabel='[mm]')
    for i, _ in enumerate(targetObjPoints):
        ax1.plot(targetObjPoints[i][0],targetObjPoints[i][1],"o",color="g")

    ax2.set_title("Image Points")
    ax2.set(xlabel='[px]', ylabel='[px]')
    ax2.imshow(frame_markers)
    for i, _ in enumerate(projImgPoints):
        ax2.plot(projImgPoints[i][0][0],projImgPoints[i][0][1],"x",color="b")
    for i, _ in enumerate(targetImgPointsSP):
        ax2.plot(targetImgPointsSP[i][0],targetImgPointsSP[i][1],"x",color="r")
    plt.show()

# Find the imgCorners in the images
def calibrationCorners(basicObjPoints, tvec, rvec, mtx, dist, search_radius, img, gray, parameter):

    # Calculate with marker pose
    tg = np.asarray(basicObjPoints)
    for i, _ in enumerate(basicObjPoints):
        v = np.dot(cv2.Rodrigues(rvec)[0], basicObjPoints[i])
        tg[i] = tvec + v

    projImgPoints, _ = cv2.projectPoints(tg,np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]), mtx, dist)

    # Delete obj/img Points that are not in the picture
    targetObjPoints = []
    targetImgPoints = []

    for i, _ in enumerate(projImgPoints):
        if projImgPoints[i][0][0] > 0 and projImgPoints[i][0][0] < img.shape[1]:
            if projImgPoints[i][0][1] > 0 and projImgPoints[i][0][1] < img.shape[0]:
                targetImgPoints.append(projImgPoints[i][0])
                targetObjPoints.append(basicObjPoints[i])

    # convert imgPoints from list to array
    targetImgPointsArray = np.array([[0., 0.]]*len(targetImgPoints),np.float32)
    for i, _ in enumerate(targetImgPoints):
        targetImgPointsArray[i] = targetImgPoints[i]

    # Corner subpixel detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, parameter['criteria']['maxIter'], parameter['criteria']['diff'])
    targetImgPointsArray = cv2.cornerSubPix(gray,targetImgPointsArray,(search_radius,search_radius),(-1,-1),criteria)

    targetImgPointsSP = []
    for i, _ in enumerate(targetImgPointsArray):
        targetImgPointsSP.append(targetImgPointsArray[i])

    # Delete not detected corners
    delIndexSP = []
    for i, _ in enumerate(targetImgPointsArray):
        if all(targetImgPointsSP[i] == targetImgPoints[i]):
            delIndexSP.append(i)
    sortedDelIndexSP = sorted(delIndexSP, reverse = True)
    for i in sortedDelIndexSP:
        del targetImgPointsSP[i]
        del targetObjPoints[i]

    return targetImgPointsSP, targetObjPoints, projImgPoints

# Calculate the corners which are hidden by the AruCo marker in the checkerboard center
def deleteAruCoHidden(gridHeight, gridWidth, parameter):
    if parameter['bottomLeftField'] == 'white':
        if (parameter['gridMaxHeight'] - gridHeight) + (parameter['gridMaxWidth'] - gridWidth) % 4 == 0:
            bottomLeftField = 0
        else:
            bottomLeftField = 1
    else:
        if (parameter['gridMaxHeight'] - gridHeight) + (parameter['gridMaxWidth'] - gridWidth) % 4 == 0:
            bottomLeftField = 1
        else:
            bottomLeftField = 0
    delIndex = []
    for i in range(gridHeight * gridWidth):
        minrow = (gridWidth-parameter['aruCoHiddenSize']-1)/2
        maxrow = (gridWidth-parameter['aruCoHiddenSize']-1)/2+parameter['aruCoHiddenSize']
        mincol = (gridHeight-parameter['aruCoHiddenSize']-1)/2
        maxcol = (gridHeight-parameter['aruCoHiddenSize']-1)/2+parameter['aruCoHiddenSize']
        if (minrow <= i % gridWidth <= maxrow) and (mincol <= int(i / gridWidth) <= maxcol):
            delIndex.append(i)
        odd = (int(i % gridWidth) + int(i / gridWidth) + bottomLeftField) % 2
        if (i % gridWidth == minrow) and (int(i / gridWidth) == mincol):
            if odd == 0:
                del delIndex[-1]
        if (i % gridWidth == minrow) and (int(i / gridWidth) == maxcol):
            if odd == 1:
                del delIndex[-1]
        if (i % gridWidth == maxrow) and (int(i / gridWidth) == mincol):
            if odd == 1:
                del delIndex[-1]
        if (i % gridWidth == maxrow) and (int(i / gridWidth) == maxcol):
            if odd == 0:
                del delIndex[-1]
    return delIndex


def calibrateCoded(parameter, img_all, gridHeight, gridWidth, mtx, dist, search_radius = 15, plot=False, oldAllObjPoints = None, oldAllImgPoints = None):

    if oldAllObjPoints is None:
        oldAllObjPoints = []

    if oldAllImgPoints is None:
        oldAllImgPoints = []

    # get grid- and AruCo-size from the parameters
    gridSize, arucoSize = parameter['gridSize'], parameter['arucoSize']

    # Calculate which checkerboard corners are missing because of the AruCo marker
    delIndex = deleteAruCoHidden(gridHeight, gridWidth, parameter)

    # Get the object points
    basicObjPoints = [0]*gridHeight*gridWidth
    for j in range(gridHeight):
        for i in range(gridWidth):
            basicObjPoints[i+gridWidth*j] = np.array([gridSize*i-gridWidth//2*gridSize,gridSize*j-gridHeight//2*gridSize,0.],np.float32)
    sortedDelIndex = sorted(delIndex, reverse = True)
    for i in sortedDelIndex:
        del basicObjPoints[i]

    allObjPoints = []
    allImgPoints = []

    # Loop over all images in this subset
    for img_num, img in enumerate(img_all):

        # Marker detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        detector = getDetector(parameter)
        corners, ids, _ = detector.detectMarkers(gray) #PARAMETER
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners)

        # If AruCo is found
        if ids!=None:

            # Use the img- and objPoints from the previous calibration to calculate the target pose
            useOldPoints = True

            # If this shouldn't be done or if there are no previous points, use the AruCo marker
            if oldAllImgPoints == [] or oldAllObjPoints == [] or useOldPoints == False:
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, arucoSize, mtx, dist)
                targetImgPointsSP, targetObjPoints, projImgPoints = calibrationCorners(basicObjPoints, tvec, rvec, mtx, dist, search_radius, img, gray, parameter)

            # If it should be done, use the old points and calculate the pose with solvePnP
            else:
                targetObjPoints = oldAllObjPoints[img_num]
                targetImgPointsSP = oldAllImgPoints[img_num]
                pose = cv2.solvePnP(np.asarray(targetObjPoints), np.asarray(targetImgPointsSP), mtx, dist, cv2.SOLVEPNP_ITERATIVE)
                tvec = np.asarray([pose[2].T])
                rvec = pose[1]
                targetImgPointsSP, targetObjPoints, projImgPoints = calibrationCorners(basicObjPoints, tvec, rvec, mtx, dist, search_radius, img, gray, parameter)

            # Append the points from this image to the list of all images
            allObjPoints.append(np.asarray(targetObjPoints))
            allImgPoints.append(np.asarray(targetImgPointsSP))

            # Plot the found points
            if plot == True:
                if img_num in parameter['plotList']:
                    plotFunc(img_num, targetObjPoints, projImgPoints, targetImgPointsSP, frame_markers)

        # Raise an error if the marker wasn't found
        else:
            print("Error: " + str(img_num) + " (No marker detected!)\n")

    # Calibrate the camera with the new Points
    rmse, mtx, dist, _ , _ = cv2.calibrateCamera(allObjPoints, allImgPoints, img.shape[0:2], mtx, dist)#, flags = cv2.CALIB_FIX_K3)

    # Print the results
    printResults(mtx, dist, rmse, gridHeight, gridWidth, search_radius)

    return rmse, mtx, dist, allObjPoints, allImgPoints
