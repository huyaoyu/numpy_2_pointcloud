#!/usr/bin/env python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

import glob
import numpy as np
import os
from pyquaternion import Quaternion
import time

import rospy
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import tf
from visualization_msgs.msg import Marker, MarkerArray

import PointCloudUtils as pcu

import CommonDefinition
from QuaternionHelpers import find_quaternion_between_vectors

NODE_NAME = "pc_normal_node"
TOPIC_NAME = "pc_normal"
ALIGNED_FRAME_ID = CommonDefinition.COMMON_ALIGNED_FRAME_ID

def publisher(files, loader, r=1, flagHaveColor=True, colorMaxDist=None, 
        nodeName=NODE_NAME, topicName=TOPIC_NAME, alignedFrame=ALIGNED_FRAME_ID):
    pub = rospy.Publisher(topicName, MarkerArray, queue_size=100)

    rospy.init_node(nodeName, anonymous=True)
    rate = rospy.Rate(r)

    time.sleep(1)

    N = len(files)

    xAxis = np.array((1,0,0), dtype=np.float32)

    countMarkerArrow = 0
    countMarkerText  = 0
    countFrame  = 0

    frameID = alignedFrame

    while ( not rospy.is_shutdown() and countFrame < N ):
        # frameID = "LIDAR_%03d" % (countFrame)
        
        f = files[countFrame]
        
        # Read the point cloud. 
        pc = loader(f)

        markers = []

        for idx in range( pc.shape[0] ):
            # Centroid.
            centroid = pc[idx, :3]

            # Normal
            normal = pc[idx, 3:6]

            # Find the quaternion.
            q = find_quaternion_between_vectors( xAxis, normal )

            # Pose.
            p = Pose()
            p.position.x = centroid[0]
            p.position.y = centroid[1]
            p.position.z = centroid[2]
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]

            # The marker.
            markerArrow = Marker( \
                header=Header(frame_id=frameID),
                ns="MarkerNormalArrow",
                id=countMarkerArrow,
                type=Marker.ARROW,
                action=0,
                pose=p,
                scale=Vector3(0.2, 0.02, 0.02),
                color=ColorRGBA(1.0, 0.0, 0.0, 0.75) )

            countMarkerArrow += 1

            markerText = Marker( \
                header=Header(frame_id=frameID),
                ns="MarkerNormalText",
                id=countMarkerText,
                type=Marker.TEXT_VIEW_FACING,
                action=0,
                pose=p,
                scale=Vector3(0.0, 0.0, 0.05),
                color=ColorRGBA(0.0, 1.0, 0.0, 0.75),
                text="N%03d" % (idx) )

            countMarkerText += 1

            # lifetime=rospy.Duration(0)
            
            markers.append( markerArrow )
            markers.append( markerText )

        # while ( not rospy.is_shutdown() ):    
        pub.publish(MarkerArray(markers=markers))
        rospy.loginfo("MarkerArray with %d markers is publised." % ( len(markers) ))

        rate.sleep()
        
        countFrame += 1

    rospy.loginfo("All normals published.")

if __name__ == "__main__":
    inDir        = rospy.get_param( "/%s/%s" % (NODE_NAME, "input_dir"      ) )
    pattern      = rospy.get_param( "/%s/%s" % (NODE_NAME, "pattern"        ) )
    ext          = rospy.get_param( "/%s/%s" % (NODE_NAME, "ext"            ) )
    rate         = rospy.get_param( "/%s/%s" % (NODE_NAME, "rate"           ) )
    color        = rospy.get_param( "/%s/%s" % (NODE_NAME, "color"          ) )
    colorMaxDist = rospy.get_param( "/%s/%s" % (NODE_NAME, "color_max_dist" ) )
    alignedFrame = rospy.get_param( "/%s/%s" % (NODE_NAME, "aligned_frame"  ), ALIGNED_FRAME_ID)

    # Find all the files.
    files = sorted( glob.glob( "%s/%s%s" % (inDir, pattern, ext) ) )

    if ( 0 == len(files) ):
        raise Exception("No files found at %s with suffix %s and ext %s. " % ( inDir, suffix, ext ))

    if ( colorMaxDist <= 0 ):
        colorMaxDist = None

    if ( ".npy" == ext ):
        loader = pcu.NumPyLoader(flagBinary=True)
    elif ( ".dat" == ext ):
        loader = pcu.NumPyLoader(flagBinary=False)
    elif ( ".csv" == ext ):
        loader = pcu.NumPyLoader(flagBinary=False, delimiter=",")
    elif ( ".ply" == ext ):
        loader = pcu.PLYLoader()
    else:
        raise Exception("Un-recognized ext: %s. " % (ext))

    try:
        # publisher(files, loader, rate, color, colorMaxDist)
        publisher(files, loader, rate, False, colorMaxDist, 
            alignedFrame=alignedFrame)
    except rospy.ROSInterruptException:
        pass
