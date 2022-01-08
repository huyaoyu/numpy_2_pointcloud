#!/usr/bin/env python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

import glob
import numpy as np
import os
from pyquaternion import Quaternion
import time

import rospy
from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import tf
from visualization_msgs.msg import Marker, MarkerArray

def publisher(fn, r=1, flagHaveColor=True, colorMaxDist=None):
    pub = rospy.Publisher("pc_camera_pose", MarkerArray, queue_size=10)

    rospy.init_node("pc_camera_pose_node", anonymous=True)
    rate = rospy.Rate(r)

    time.sleep(1)

    qX2Z = Quaternion( axis=(0,1,0), radians=-np.pi/2 )

    countMarkerCamera = 0
    countMarkerText   = 0

    frameID = "LIDAR_%03d" % (0)
    
    # Read the point cloud. 
    pc = np.loadtxt(fn, delimiter=",").astype(np.float32)

    markers = []

    for idx in range( pc.shape[0] ):
        if ( rospy.is_shutdown() ):
            break

        # Centroid.
        centroid = pc[idx, 5:8]

        # Quaternion
        q = pc[idx, 1:5]

        qC = Quaternion( w=q[0], x=q[1], y=q[2], z=q[3] )

        # Combined quaternion.
        qZ = qC * qX2Z

        # Pose.
        p = Pose()
        p.position.x = centroid[0]
        p.position.y = centroid[1]
        p.position.z = centroid[2]
        p.orientation.x = qZ[1]
        p.orientation.y = qZ[2]
        p.orientation.z = qZ[3]
        p.orientation.w = qZ[0] # This is w.

        # The marker.
        markerArrow = Marker( \
            header=Header(frame_id=frameID),
            ns="MarkerCameraPose",
            id=countMarkerCamera,
            type=Marker.ARROW,
            action=0,
            pose=p,
            scale=Vector3(0.2, 0.02, 0.02),
            color=ColorRGBA(0.0, 0.0, 1.0, 0.75) )

        countMarkerCamera += 1

        markerText = Marker( \
            header=Header(frame_id=frameID),
            ns="MarkerCameraPoseName",
            id=countMarkerText,
            type=Marker.TEXT_VIEW_FACING,
            action=0,
            pose=p,
            scale=Vector3(0.0, 0.0, 0.05),
            color=ColorRGBA(0.0, 1.0, 0.0, 0.75),
            text="C%04d" % (int(pc[idx, 0])) )

        countMarkerText += 1

        # lifetime=rospy.Duration(0)
        
        markers.append(markerArrow)
        markers.append(markerText)

    markerArray = MarkerArray(markers=markers)

    # while (not rospy.is_shutdown()):
    pub.publish(markerArray)
    rospy.loginfo("MarkerArray published with %d markers." % (len(markers)))

    rate.sleep()

    rospy.loginfo("All camera poses published.")

if __name__ == "__main__":
    inFile   = rospy.get_param("/pc_camera_pose_node/infile")
    rate     = rospy.get_param("/pc_camera_pose_node/rate")
    color    = rospy.get_param("/pc_camera_pose_node/color")
    colorMaxDist = rospy.get_param("/pc_camera_pose_node/color_max_dist")

    try:
        # publisher(files, loader, rate, color, colorMaxDist)
        publisher(inFile, rate, False, colorMaxDist)
    except rospy.ROSInterruptException:
        pass
