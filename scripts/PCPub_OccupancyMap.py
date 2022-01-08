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

NODE_NAME = "pc_occ_map_node"
TOPIC_NAME = "pc_occ_map"
ALIGNED_FRAME_ID = CommonDefinition.COMMON_ALIGNED_FRAME_ID

OCP_MAP_OCC_FREE     = 0
OCP_MAP_OCC_OCCUPIED = 1
OCP_MAP_OCC_FRONTIER = 3

VOXEL_SIZE = [ 0.05, 0.05, 0.05 ]

def get_filename_parts(fn):
    p = os.path.split(fn)

    if ( "" == p[0] ):
        p = (".", p[1])

    f = os.path.splitext(p[1])

    return [ p[0], f[0], f[1] ]

def publisher(fn, r=1, flagHaveColor=True, \
        nodeName=NODE_NAME, topicName=TOPIC_NAME, alignedFrame=ALIGNED_FRAME_ID):
    pub  = rospy.Publisher(topicName, Marker, queue_size=10)
    pubF = rospy.Publisher("%s_%s" % (topicName, "f"), Marker, queue_size=10)

    rospy.init_node(nodeName, anonymous=True)
    rate = rospy.Rate(r)

    time.sleep(1)

    frameID = alignedFrame
    
    # Read the point coordinates from the CSV file.
    pc = np.loadtxt(fn, delimiter=",")
    rospy.loginfo("%d points loaded. " % ( pc.shape[0] ) )

    cubesOccupied = []
    cubesFree     = []
    cubesFrontier = []

    for idx in range( pc.shape[0] ):
        if ( rospy.is_shutdown() ):
            break

        # Centroid.
        centroid = pc[idx, :3]

        # Mask.
        m = pc[idx, 3]

        if ( OCP_MAP_OCC_FREE == m ):
            cubesFree.append( Point( centroid[0], centroid[1], centroid[2] ) )
        elif ( OCP_MAP_OCC_OCCUPIED == m ):
            cubesOccupied.append( Point( centroid[0], centroid[1], centroid[2] ) )
        elif ( OCP_MAP_OCC_FRONTIER == m ):
            cubesFrontier.append( Point( centroid[0], centroid[1], centroid[2] ) )

    countFree     = len( cubesFree )
    countOccupied = len( cubesOccupied )
    countFrontier = len( cubesFrontier )

    # Pose.
    p = Pose()
    p.position.x = 0.0
    p.position.y = 0.0
    p.position.z = 0.0
    p.orientation.x = 0.0
    p.orientation.y = 0.0
    p.orientation.z = 0.0
    p.orientation.w = 1.0 

    # The marker.
    if ( countFree > 0 ):
        markerCubeList = Marker( \
            header=Header(frame_id=frameID),
            ns="OccupancyCube",
            id=0,
            type=Marker.CUBE_LIST,
            action=0,
            pose=p,
            scale=Vector3(VOXEL_SIZE[0], VOXEL_SIZE[1], VOXEL_SIZE[2]),
            color=ColorRGBA(0.0, 1.0, 0.0, 1.0), 
            points=cubesFree )

        pub.publish(markerCubeList)
        rospy.loginfo("%d free voxels." % (countFree))
        rate.sleep()
    
    if ( countOccupied > 0 ):
        markerCubeList = Marker( \
            header=Header(frame_id=frameID),
            ns="OccupancyCube",
            id=1,
            type=Marker.CUBE_LIST,
            action=0,
            pose=p,
            scale=Vector3(VOXEL_SIZE[0], VOXEL_SIZE[1], VOXEL_SIZE[2]),
            color=ColorRGBA(1.0, 0.0, 0.0, 1.0),
            points=cubesOccupied )

        pub.publish(markerCubeList)
        rospy.loginfo("%d occupied voxels." % (countOccupied))
        rate.sleep()
    
    if ( countFrontier > 0 ):
        markerCubeList = Marker( \
            header=Header(frame_id=frameID),
            ns="MarkerPointCube",
            id=3,
            type=Marker.CUBE_LIST,
            action=0,
            pose=p,
            scale=Vector3(VOXEL_SIZE[0], VOXEL_SIZE[1], VOXEL_SIZE[2]),
            color=ColorRGBA(0.0, 0.0, 1.0, 0.75),
            points=cubesFrontier )

        while ( not rospy.is_shutdown() ):
            pubF.publish(markerCubeList)
            rospy.loginfo("%d frontier voxels." % (countFrontier))
            rate.sleep()

if __name__ == "__main__":
    inFile       = rospy.get_param( "/%s/%s" % (NODE_NAME,"infile") )
    rate         = rospy.get_param( "/%s/%s" % (NODE_NAME,"rate"  ) )
    color        = rospy.get_param( "/%s/%s" % (NODE_NAME,"color" ) )
    alignedFrame = rospy.get_param( "/%s/%s" % (NODE_NAME, "aligned_frame" ), ALIGNED_FRAME_ID)

    try:
        publisher(inFile, rate, False, alignedFrame=alignedFrame)
    except rospy.ROSInterruptException:
        pass
