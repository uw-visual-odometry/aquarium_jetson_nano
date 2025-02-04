import cv2

# This is fixed for these cameras

image_size = (1080,1440)
framerate = 30

# GStreamer string, may be possible to optimize this further?
gst_string = '''
nvarguscamerasrc sensor-id={camera_num} wbmode=0 aelock=true ispdigitalgainrange=\"1 8\" gainrange=\"1 48\" ! 
    video/x-raw(memory:NVMM),width={image_size[1]},height={image_size[0]},framerate={framerate}/1 ! nvvidconv !
    queue ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink
'''

cap = cv2.VideoCapture(gst_string.format(camera_num=0, image_size=image_size, framerate=framerate), cv2.CAP_GSTREAMER )

# Open OpenCV camera
#    left = cv2.VideoCapture(gst_string.format(camera_num=0, image_size=image_size, #framerate=framerate), cv2.CAP_GSTREAMER )
#    right = cv2.VideoCapture(gst_string.format(camera_num=1, image_size=image_size, #framerate=framerate), cv2.CAP_GSTREAMER )



import numpy as np
import cv2 as cv
import glob
import pickle



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,6)
frameSize = (640,480)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


num = 0
while cap.isOpened():

    succes, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    cv.imshow('img', gray)
    cv.waitKey()


    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey()



    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
