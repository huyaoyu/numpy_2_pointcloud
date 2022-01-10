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

import CameraDescriptor as CD
import CommonDefinition
from CommonUtils import get_filename_parts

NODE_NAME = "pc_camera_pose_node"
TOPIC_NAME = "pc_camera_pose"
ALIGNED_FRAME_ID = CommonDefinition.COMMON_ALIGNED_FRAME_ID

def read_cameras_as_camera_descriptors(fn):
    parts = get_filename_parts(fn)

    if ( ".csv" == parts[2] ):
        return CD.read_cam_proj_csv(fn)
    elif ( ".json" == parts[2] ):
        return CD.read_cam_proj_json(fn)
    else:
        raise Exception("Unsupported file extension {} of file. ".format(parts[2], fn))

def publisher(fn, r=1, flagHaveColor=True, colorMaxDist=None, \
        nodeName=NODE_NAME, topicName=TOPIC_NAME, alignedFrame=ALIGNED_FRAME_ID):
    pub = rospy.Publisher(topicName, MarkerArray, queue_size=10)

    rospy.init_node(nodeName, anonymous=True)
    rate = rospy.Rate(r)

    time.sleep(1)

    qX2Z = Quaternion( axis=(0,1,0), radians=-np.pi/2 )

    countMarkerCamera = 0
    countMarkerText   = 0

    frameID = alignedFrame
    
    # Read the camera pose as CameraDescriptor objects.
    camDes = read_cameras_as_camera_descriptors(fn)
    rospy.loginfo("%d cameras loaded. " % ( len(camDes) ) )

    markers = []

    for idx in range( len(camDes) ):
        if ( rospy.is_shutdown() ):
            break

        # Centroid.
        centroid = camDes[idx].get_centroid()

        # Quaternion
        qC = camDes[idx].get_quaternion()

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
            text="C%04d" % (camDes[idx].get_id()) )

        countMarkerText += 1

        # lifetime=rospy.Duration(0)
        
        markers.append(markerArrow)
        markers.append(markerText)

    markerArray = MarkerArray(markers=markers)

    while (not rospy.is_shutdown()):
        pub.publish(markerArray)
        rospy.loginfo("MarkerArray published with %d markers." % (len(markers)))

        rate.sleep()

        rospy.loginfo("All camera poses published.")

if __name__ == "__main__":
    rate         = rospy.get_param( "/%s/%s" % (NODE_NAME, "rate"          ) )
    inFile       = rospy.get_param( "/%s/%s" % (NODE_NAME, "infile"        ) )
    colorMaxDist = rospy.get_param( "/%s/%s" % (NODE_NAME, "color_max_dist") )
    color        = rospy.get_param( "/%s/%s" % (NODE_NAME, "color"         ) )
    alignedFrame = rospy.get_param( "/%s/%s" % (NODE_NAME, "aligned_frame" ), ALIGNED_FRAME_ID)

    try:
        # publisher(files, loader, rate, color, colorMaxDist)
        publisher(inFile, rate, False, colorMaxDist)
    except rospy.ROSInterruptException:
        pass
