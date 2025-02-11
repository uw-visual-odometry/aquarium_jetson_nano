import numpy as np
import glob
import cv2

image_size = (540,720)
framerate = 30
# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


# Grab left and right images
imagesL = glob.glob('/home/sysop/aquarium_jetson_nano/camera_calib/images/both/left/*.png')
imagesL.sort()
imagesR = glob.glob('/home/sysop/aquarium_jetson_nano/camera_calib/images/both/right/*.png')
imagesR.sort()

idx = 0

for imageL in imagesL:
    print(imageL)
    img = cv2.imread(imageL)
    cv2.imwrite('../caliImages/stereo/l/caliResult_' + str(idx) + '_0.png', img)
    
    img = cv2.remap(img, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    cv2.imwrite('../caliImages/stereo/l/caliResult_' + str(idx) + '_1.png', img)
    
    idx += 1

idx = 0

for imageR in imagesR:
    print(imageR)
    img = cv2.imread(imageR)
    cv2.imwrite('../caliImages/stereo/r/caliResult_' + str(idx) + '_0.png', img)
    
    img = cv2.remap(img, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    cv2.imwrite('../caliImages/stereo/r/caliResult_' + str(idx) + '_1.png', img)
    
    idx += 1
    
