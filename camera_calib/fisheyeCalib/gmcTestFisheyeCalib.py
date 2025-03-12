import numpy as np
import cv2 as cv
import glob

####@@@@#### Detect chessboard corners ####@@@@####

CHECKERBOARD = (10,7)
frameSize = (1440, 1080)
# calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_CHECK_COND + cv.fisheye.CALIB_FIX_SKEW
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW

# termination criteria
# term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# 1
# objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
# objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
# 2
# objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float64)
# objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# 3
objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# size_of_chessboard_squares_mm = 20
# objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imagesLeft, imagesRight = [], []

#imagesLeft += sorted(glob.glob('../bothImages/left2/*.png'))
#imagesLeft += sorted(glob.glob('../bothImages/left3/*.png'))
#imagesLeft += sorted(glob.glob('../bothImages/left4/*.png'))
#imagesLeft += sorted(glob.glob('../bothImages/left5/*.png'))
#imagesLeft += sorted(glob.glob('../bothImages/left6/*.png'))
imagesLeft += sorted(glob.glob('../bothImages/left7/*.png'))
#imagesRight += sorted(glob.glob('../bothImages/right2/*.png'))
#imagesRight += sorted(glob.glob('../bothImages/right3/*.png'))
#imagesRight += sorted(glob.glob('../bothImages/right4/*.png'))
#imagesRight += sorted(glob.glob('../bothImages/right5/*.png'))
#imagesRight += sorted(glob.glob('../bothImages/right6/*.png'))
imagesRight += sorted(glob.glob('../bothImages/right7/*.png'))
for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    # retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retL, cornersL = cv.findChessboardCorners(grayL, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
    # retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
    
    # If found, add object points, image points (after refining them)
    if retL and retR == True:
   

        objpoints.append(objp)

        # cornersL = cv.cornerSubPix(grayL, cornersL, (12,11), (-1,-1), term_criteria)
        cornersL = cv.cornerSubPix(grayL, cornersL, (3,3), (-1,-1), subpix_criteria)
        imgpointsL.append(cornersL)

        # cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), term_criteria)
        cornersR = cv.cornerSubPix(grayR, cornersR, (3,3), (-1,-1), subpix_criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        # cv.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
        # cv.imshow('img left', imgL)
        # cv.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)
        # cv.imshow('img right', imgR)
        # cv.waitKey(1)
    else:
    	print(imgLeft)

cv.destroyAllWindows()
####@@@@#### CALIBRATION ####@@@@####

N_OK = len(objpoints)
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

R = np.zeros((1, 1, 3), dtype=np.float64)
T = np.zeros((1, 1, 3), dtype=np.float64)

K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))

rms, _, _, _, _ = \
    cv.fisheye.calibrate(
        objpoints,
        imgpointsL,
        grayL.shape[::-1],
        K_left,
        D_left,
        rvecs,
        tvecs,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))

rms, _, _, _, _ = \
    cv.fisheye.calibrate(
        objpoints,
        imgpointsR,
        grayL.shape[::-1],
        K_right,
        D_right,
        rvecs,
        tvecs,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print(len(imagesLeft))
print("Found " + str(N_OK) + " valid images for calibration")
print("RMS=" + str(rms))
print("DIM=" + str(frameSize[::-1]))
print("K_left=np.array(" + str(K_left.tolist()) + ")")
print("D_left=np.array(" + str(D_left.tolist()) + ")")
print("K_right=np.array(" + str(K_right.tolist()) + ")")
print("D_right=np.array(" + str(D_right.tolist()) + ")")

print("calibrating both fisheye lenses")

####@@@@#### Stereo Calibration ####@@@@####

objpoints = np.array([objp]*len(imgpointsL), dtype=np.float64)
imgpointsL = np.asarray(imgpointsL, dtype=np.float64)
imgpointsR = np.asarray(imgpointsR, dtype=np.float64)

objpoints = np.reshape(objpoints, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))
imgpointsL = np.reshape(imgpointsL, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))
imgpointsR = np.reshape(imgpointsR, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))

(rms, K1, D1, K2, D2, R, T) = cv.fisheye.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    K_left,
    D_left,
    K_right,
    D_right,
    grayL.shape[::-1],
    R,
    T,
    calibration_flags
)

print("\nSTEREO RMS=" + str(rms))
print("K1=np.array(" + str(K1.tolist()) + ")")
print("D1=np.array(" + str(D1.tolist()) + ")")
print("K2=np.array(" + str(K2.tolist()) + ")")
print("D2=np.array(" + str(D2.tolist()) + ")")
print("Rotation=np.array(" + str(R.tolist()) + ")")
print("Translation=np.array(" + str(T.tolist()) + ")")

# retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
# heightL, widthL, channelsL = imgL.shape
# newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

# retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
# heightR, widthR, channelsR = imgR.shape
# newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

# flags = 0
# flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

# criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
# retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

####@@@@#### Stereo Rectification ####@@@@####

R1 = np.zeros([3,3])
R2 = np.zeros([3,3])
P1 = np.zeros([3,4])
P2 = np.zeros([3,4])
Q = np.zeros([4,4])

rectL, rectR, projMatrixL, projMatrixR, Q = cv.fisheye.stereoRectify(
    K1,
    D1,
    K2,
    D2,
    grayL.shape[::-1],
    R,
    T,
    cv.fisheye.CALIB_ZERO_DISPARITY,
    # balance=0.5,
    # fov_scale=0.6
    R2, P1, P2, Q,
    cv.CALIB_ZERO_DISPARITY, (0,0), 0, 0
)

print("Saving!")
print('rectL:')
print(rectL)
print('rectR:')
print(rectR)
print('projMatrixL:')
print(projMatrixL)
print('projMatrixR:')
print(projMatrixR)
print('Q:')
print(Q)

stereoMapL = cv.fisheye.initUndistortRectifyMap(K_left, D_left, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.fisheye.initUndistortRectifyMap(K_right, D_right, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving!")

cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()

cv_file2 = cv.FileStorage("totalParams.txt", cv.FILE_STORAGE_WRITE)
cv_file2.write('K1',K1)
cv_file2.write('K2',K2)
cv_file2.write('D1',D1)
cv_file2.write('D2',D2)
cv_file2.write('rectL',rectL)
cv_file2.write('rectR',rectR)
cv_file2.write('projMatrixL',projMatrixL)
cv_file2.write('projMatrixR',projMatrixR)
cv_file2.write('Q',Q)

cv_file2.release()

