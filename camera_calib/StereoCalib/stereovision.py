import numpy as np
import cv2

image_size = (540,720)
image_size = (480,640)

framerate = 30
# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


# Open both cameras
# cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
# cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

gst_string = '''
nvarguscamerasrc sensor-id={camera_num} wbmode=0 aelock=true ispdigitalgainrange=\"1 8\" gainrange=\"1 48\" ! 
    video/x-raw(memory:NVMM),width={image_size[1]},height={image_size[0]},framerate={framerate}/1 ! nvvidconv !
    queue ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink
'''
cap_left = cv2.VideoCapture(gst_string.format(camera_num=0, image_size=image_size, framerate=framerate), cv2.CAP_GSTREAMER )
cap_right = cv2.VideoCapture(gst_string.format(camera_num=1, image_size=image_size, framerate=framerate), cv2.CAP_GSTREAMER )


while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                     
    # Show the frames
    cv2.imshow("frame right", frame_right) 
    cv2.imshow("frame left", frame_left)


    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
