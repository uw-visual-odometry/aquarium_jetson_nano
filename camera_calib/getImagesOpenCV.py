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

num = 0
print(cap.isOpened())
while cap.isOpened():

    succes, img = cap.read()
    
   

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
