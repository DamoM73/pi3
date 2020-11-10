# import necessary packages
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

# define the lower and upper boundaries of the ball in the HSV color space
blue_lower = (89,138,50)
blue_upper = (125,255,164)

# grab the reference to the webcam
vs = VideoStream(src=0, usePiCamera=True).start()
    
# allow the camera or video file to intialise
time.sleep(0.5)

# main loop
while True:
    # grab the current frame
    frame = vs.read()
    
    # resize the frame, blur it flip it, and convert it to the HSV color space
    frame = cv2.flip(frame, -1)
    frame = imutils.resize(frame, width=1200)
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color 'blue', then perform a series of dilations and erosions
    # to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find the contours in the mask and initialize the current (x,y) center of the ball
    cents = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cents = imutils.grab_contours(cents)
    center = None
    
    # only proceed if at least one contour was found
    if len(cents) > 0:
        # find the largest contour in the mask,then use it to compute 
        # the minimum enclosing circle and centroid
        c = max(cents, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        centre = (int(M["m10"] / M["m00"]), int(M["m01"]/M["m00"]))
        
        # only proceed if the radius meets minimum size
        if radius > 10:
            # draw the circle and centroid on the frame, then update the list of the tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),(0,255,255),2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
      
    # show frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
# release the camera
vs.stop()
    
# close all windows
cv2.destroyAllWindows()
