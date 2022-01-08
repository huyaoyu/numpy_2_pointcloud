#!/usr/bin/env python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

import glob
import json
import numpy as np
import os
from plyfile import PlyData
from pyquaternion import Quaternion
import time

import rospy
from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import tf
from visualization_msgs.msg import Marker, MarkerArray

from ColorMapping import hex_2_RGBA

import CommonDefinition
from QuaternionHelpers import find_quaternion_between_vectors

NODE_NAME = "pc_hole_polygon_node"
TOPIC_NAME = "pc_hole_polygon"
ALIGNED_FRAME_ID = CommonDefinition.COMMON_ALIGNED_FRAME_ID

def convert_numpy_table_2_ros_point(table):
    if ( table.shape[1] != 3 ):
        raise Exception("table.shape = {}".format(table.shape))

    points = []

    for i in range( table.shape[0] ):
        points.append( 
            Point( table[i,0], table[i,1], table[i,2] ))

    return points

def publisher(fnCloud, fnPolygon, r=1, 
        colorCamString="#228C22BF", colorNoCamString="#FF7400BF", 
        nodeName=NODE_NAME, topicName=TOPIC_NAME, alignedFrame=ALIGNED_FRAME_ID, 
        flagPubNormalAndID=False):
    pub = rospy.Publisher(topicName, MarkerArray, queue_size=10)

    rospy.init_node(nodeName, anonymous=True)
    rate = rospy.Rate(r)

    time.sleep(1)

    countMarkerHolePolygon = 0
    countMarkerText   = 0

    frameID = alignedFrame

    # For the normal.
    xAxis = np.array((1,0,0), dtype=np.float32)

    # Color.
    colorCam   = hex_2_RGBA(colorCamString)
    colorNoCam = hex_2_RGBA(colorNoCamString)
    rgbaCam    = ColorRGBA(
        colorCam[0]/255.0, 
        colorCam[1]/255.0, 
        colorCam[2]/255.0, 
        colorCam[3]/255.0)
    rgbaNoCam  = ColorRGBA(
        colorNoCam[0]/255.0, 
        colorNoCam[1]/255.0, 
        colorNoCam[2]/255.0, 
        colorNoCam[3]/255.0)
    
    # Read the point cloud. 
    plydata = PlyData.read(fnCloud)
    inCloud = np.stack( ( 
        plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]
     ), axis=1 )

    # Read the JSON file.
    with open(fnPolygon) as fp:
        hpp = json.load(fp)

    markers = []

    for idx in range( len( hpp["hpp"] ) ):
        if ( rospy.is_shutdown() ):
            break

        # Polygon indices.
        indices = hpp["hpp"][idx]["polygonIndices"]

        # The points.
        coordinates = inCloud[ indices, : ]
        points = convert_numpy_table_2_ros_point(coordinates)

        points.append( points[0] )

        # Pose.
        pose = Pose()
        pose.position.x = 0
        pose.position.y = 0
        pose.position.z = 0
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1

        # Color.
        if ( hpp["hpp"][idx]["haveCamProj"] ):
            polygonColor = rgbaCam
        else:
            polygonColor = rgbaNoCam

        # The marker.
        markerLineStrip = Marker( \
            header=Header(frame_id=frameID),
            ns="MarkerHolePolygon",
            id=countMarkerHolePolygon,
            type=Marker.LINE_STRIP,
            action=0,
            pose=pose,
            scale=Vector3(0.02, 0.00, 0.00),
            color=polygonColor,
            points=points )

        # markerText = Marker( \
        #     header=Header(frame_id=frameID),
        #     ns="MarkerHolePolygonName",
        #     id=countMarkerText,
        #     type=Marker.TEXT_VIEW_FACING,
        #     action=0,
        #     pose=p,
        #     scale=Vector3(0.0, 0.0, 0.05),
        #     color=ColorRGBA(0.0, 1.0, 0.0, 0.75),
        #     text="C%04d" % (int(pc[idx, 0])) )

        # countMarkerText += 1
        
        markers.append(markerLineStrip)
        
        if ( flagPubNormalAndID ):
            centroid = hpp["hpp"][idx]["centroid"]
            normal   = hpp["hpp"][idx]["normal"]
            q = find_quaternion_between_vectors( xAxis, normal )
            
            # Pose.
            pNormal = Pose()
            pNormal.position.x = centroid[0]
            pNormal.position.y = centroid[1]
            pNormal.position.z = centroid[2]
            pNormal.orientation.x = q[0]
            pNormal.orientation.y = q[1]
            pNormal.orientation.z = q[2]
            pNormal.orientation.w = q[3]

            markerArrow = Marker( \
                header=Header(frame_id=frameID),
                ns="MarkerHolePolygonNormal",
                id=countMarkerHolePolygon,
                type=Marker.ARROW,
                action=0,
                pose=pNormal,
                scale=Vector3(0.2, 0.02, 0.02),
                color=ColorRGBA(1.0, 0.0, 0.0, 0.75) )

            markerText = Marker( \
                header=Header(frame_id=frameID),
                ns="MarkerHolePolygonText",
                id=countMarkerHolePolygon,
                type=Marker.TEXT_VIEW_FACING,
                action=0,
                pose=pNormal,
                scale=Vector3(0.0, 0.0, 0.05),
                color=ColorRGBA(0.0, 1.0, 0.0, 0.75),
                text="N%03d" % (hpp["hpp"][idx]["id"]) )

            markers.append(markerArrow)
            markers.append(markerText)

        countMarkerHolePolygon += 1

    markerArray = MarkerArray(markers=markers)

    # while (not rospy.is_shutdown()):
    pub.publish(markerArray)
    rospy.loginfo("MarkerArray published with %d markers." % (len(markers)))

    rate.sleep()

    rospy.loginfo("All polygons published.")

if __name__ == "__main__":
    inCloud        = rospy.get_param( "/%s/%s" % ( NODE_NAME, "incloud"          ) )
    inPolygon      = rospy.get_param( "/%s/%s" % ( NODE_NAME, "inpolygon"        ) )
    rate           = rospy.get_param( "/%s/%s" % ( NODE_NAME, "rate"             ) )
    colorCam       = rospy.get_param( "/%s/%s" % ( NODE_NAME, "color_cam"        ) )
    colorNoCam     = rospy.get_param( "/%s/%s" % ( NODE_NAME, "color_no_cam"     ) )
    alignedFrame   = rospy.get_param( "/%s/%s" % ( NODE_NAME, "aligned_frame"    ), ALIGNED_FRAME_ID)
    flagNormalText = rospy.get_param( "/%s/%s" % ( NODE_NAME, "flag_normal_text" ) )

    try:
        publisher(inCloud, inPolygon, rate, colorCam, colorNoCam, 
            alignedFrame=alignedFrame, flagPubNormalAndID=flagNormalText)
    except rospy.ROSInterruptException:
        pass
