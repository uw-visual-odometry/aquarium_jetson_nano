#!/usr/bin/bash
#
# Uses the "glstereomix" GStreamer element to stream two nvargus cameras,
# combine them into a left-and-right composite image
# and stream it over RTSP to: rtsp://localhost:8554/stereo
#
# This asumes a mediamtx instance is already running on the nano:
#
#   docker run --rm -it --network=host bluenviron/mediamtx:latest
#
# It can then be played on a desktop with:
#
#   ffplay -rtsp_transport udp rtsp://<ip address of nano>:8554/stereo
#ffplay -rtsp_transport udp rtsp://<ip address of nano>:8554/stereo
#qtmux ! filesink location=test_stereo.mp4 -e

. imx296_constants.sh

gst-launch-1.0 nvarguscamerasrc sensor-id=0 name=left \
               nvarguscamerasrc sensor-id=1 name=right \
               glstereomix name=mix \
    left. ! "video/x-raw(memory:NVMM),width=$IMG_WIDTH,height=$IMG_HEIGHT,framerate=$IMG_RATE/1" ! \
            nvvidconv ! 'video/x-raw' ! glupload ! mix. \
    right. ! "video/x-raw(memory:NVMM),width=$IMG_WIDTH,height=$IMG_HEIGHT,framerate=$IMG_RATE/1" ! \
             nvvidconv !'video/x-raw' ! glupload ! mix. \
    mix. ! video/x-raw'(memory:GLMemory)',multiview-mode=side-by-side ! \
    glcolorconvert ! gldownload ! queue ! fpsdisplaysink
    

#    x264enc speed-preset=veryfast tune=zerolatency ! \
#    h264parse ! \
#    rtspclientsink location=rtsp://localhost:8554/stereo
    
    
   
