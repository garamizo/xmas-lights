#!/usr/bin/env python3
import rospy
import message_filters
from gaiteyes.msg import SerialPacket
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import yaml 
import cv2
import csv

detections = []

def image_callback(msg, args):
    global detections
    bridge, backSub, detector = args

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    fgMask = backSub.apply(cv_image)
    keypoints = detector.detect(fgMask)

    if len(keypoints) > 0:
        i = np.argmax([d.size for d in keypoints])
        time = msg.header.stamp.to_sec()
        x, y, s = keypoints[i].pt[0], keypoints[i].pt[1], keypoints[i].size
        detections.append([time, x, y, s, len(keypoints)])

    # blobs = cv2.drawKeypoints(fgMask, keypoints, np.array([]), (0,255,255), 
    #     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.putText(blobs, "Detections: %d" % (len(keypoints),), (100, 100),
    #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # cv2.imshow('Frame', blobs)
    # cv2.imshow('Original', cv_image)
    # cv2.waitKey(1)


if __name__ == '__main__':

    # cv2.namedWindow('Frame')
    # cv2.namedWindow('Original')

    bridge = CvBridge()
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 64
    params.maxThreshold = 256
    params.minDistBetweenBlobs = 200.0
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False
    params.filterByCircularity = False
    params.filterByArea = False
    params.minArea = 10.0
    params.maxArea = 1e10

    detector = cv2.SimpleBlobDetector_create(params)

    rospy.init_node('detector', anonymous=True)
    rospy.Subscriber("/stereo/left/image_raw", Image, image_callback, 
        (bridge, backSub, detector), queue_size=100)

    with open('detections.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['time', 'x', 'y', 'diameter', 'numberOfDetections']) 

        while not rospy.is_shutdown():

            if len(detections) > 0:
                det = detections.pop(0)  # (time, x, y)
                csvwriter.writerow(det) 
        
        cv2.destroyAllWindows()