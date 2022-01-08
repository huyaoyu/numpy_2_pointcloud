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

import CommonDefinition

NODE_NAME = "pc_point_cube_node"
TOPIC_NAME = "pc_point_cube"
ALIGNED_FRAME_ID = CommonDefinition.COMMON_ALIGNED_FRAME_ID

def get_filename_parts(fn):
    p = os.path.split(fn)

    if ( "" == p[0] ):
        p = (".", p[1])

    f = os.path.splitext(p[1])

    return [ p[0], f[0], f[1] ]

def publisher(fn, r=1, flagHaveColor=True, \
        nodeName=NODE_NAME, topicName=TOPIC_NAME, alignedFrame=ALIGNED_FRAME_ID):
    pub = rospy.Publisher(topicName, MarkerArray, queue_size=10)

    rospy.init_node(nodeName, anonymous=True)
    rate = rospy.Rate(r)

    time.sleep(1)

    countMarkerCube = 0
    countMarkerText = 0

    frameID = alignedFrame
    
    # Read the point coordinates from the CSV file.
    pc = np.loadtxt(fn, delimiter=",")
    rospy.loginfo("%d points loaded. " % ( pc.shape[0] ) )

    markers = []

    for idx in range( pc.shape[0] ):
        if ( rospy.is_shutdown() ):
            break

        # Centroid.
        centroid = pc[idx, :]

        # Pose.
        p = Pose()
        p.position.x = centroid[0]
        p.position.y = centroid[1]
        p.position.z = centroid[2]
        p.orientation.x = 0.0
        p.orientation.y = 0.0
        p.orientation.z = 0.0
        p.orientation.w = 1.0 

        # The marker.
        markerArrow = Marker( \
            header=Header(frame_id=frameID),
            ns="MarkerPointCube",
            id=countMarkerCube,
            type=Marker.CUBE,
            action=0,
            pose=p,
            scale=Vector3(0.2, 0.2, 0.2),
            color=ColorRGBA(1.0, 0.0, 0.0, 0.75) )

        countMarkerCube += 1

        # markerText = Marker( \
        #     header=Header(frame_id=frameID),
        #     ns="MarkerPointCubeName",
        #     id=countMarkerText,
        #     type=Marker.TEXT_VIEW_FACING,
        #     action=0,
        #     pose=p,
        #     scale=Vector3(0.0, 0.0, 0.05),
        #     color=ColorRGBA(0.0, 1.0, 0.0, 0.75),
        #     text="C%04d" % (camDes[idx].get_id()) )

        # countMarkerText += 1

        # lifetime=rospy.Duration(0)
        
        markers.append(markerArrow)
        # markers.append(markerText)

    markerArray = MarkerArray(markers=markers)

    # while (not rospy.is_shutdown()):
    pub.publish(markerArray)
    rospy.loginfo("MarkerArray published with %d markers." % (len(markers)))

    rate.sleep()

    rospy.loginfo("All camera poses published.")

if __name__ == "__main__":
    inFile       = rospy.get_param( "/%s/%s" % (NODE_NAME,"infile") )
    rate         = rospy.get_param( "/%s/%s" % (NODE_NAME,"rate"  ) )
    color        = rospy.get_param( "/%s/%s" % (NODE_NAME,"color" ) )
    alignedFrame = rospy.get_param( "/%s/%s" % (NODE_NAME, "aligned_frame" ), ALIGNED_FRAME_ID)

    try:
        publisher(inFile, rate, False, alignedFrame=alignedFrame)
    except rospy.ROSInterruptException:
        pass
