import cv2
import os


image_size = (1080,1440)
framerate = 30
#cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(2)
gst_string = '''
nvarguscamerasrc sensor-id={camera_num} wbmode=0 aelock=true ispdigitalgainrange=\"1 8\" gainrange=\"1 48\" ! 
    video/x-raw(memory:NVMM),width={image_size[1]},height={image_size[0]},framerate={framerate}/1 ! nvvidconv !
    queue ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink
'''

capL = cv2.VideoCapture(gst_string.format(camera_num=0, image_size=image_size, framerate=framerate), cv2.CAP_GSTREAMER )
capR = cv2.VideoCapture(gst_string.format(camera_num=1, image_size=image_size, framerate=framerate), cv2.CAP_GSTREAMER )
num = 0
folder_num = 7

left_dir = f"../bothImages/left{folder_num}/"
right_dir = f"../bothImages/right{folder_num}/"
if not os.path.exists(left_dir):
    os.makedirs(left_dir)
if not os.path.exists(right_dir):
    os.makedirs(right_dir)


while capL.isOpened() and capR.isOpened():

    succesL, left = capL.read()
    succesR, right = capR.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(left_dir + 'l' + str(num) + '.png', left)
        cv2.imwrite(right_dir + 'r' + str(num) + '.png', right)
        print("images saved!")
        num += 1



    cv2.imshow('Left Camera',left)
    cv2.imshow('Right Camera',right)

# Release and destroy all windows before termination
capL.release()
capR.release()

cv2.destroyAllWindows()
