#!/usr/bin/env python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

import glob
import numpy as np
import os
from pyquaternion import Quaternion
import time

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf

import CommonDefinition
import PointCloudUtils as pcu

NODE_NAME = "pc_static_node"
TOPIC_NAME = "pc_static"
ALIGNED_FRAME_ID = CommonDefinition.COMMON_ALIGNED_FRAME_ID

def publisher(files, loader, r=1, \
        flagHaveColor=True, colorMaxDist=None, \
        alignedFrameID=ALIGNED_FRAME_ID, \
        nodeName=NODE_NAME, topicName=TOPIC_NAME):
    pub  = rospy.Publisher(topicName, PointCloud2, queue_size=1)
    br   = tf.TransformBroadcaster()

    rospy.init_node(nodeName, anonymous=True)
    rate = rospy.Rate(r)

    rate.sleep()

    count = 0

    N = len(files)

    RAW = np.array([ 1, 0, 0, 0, -1, 0, 0, 0, -1 ], dtype=np.float32).reshape((3, 3))
    qAW = Quaternion(matrix=RAW)

    RLC = np.array([ 0, 1, 0, 1, 0, 0, 0, 0, -1 ], dtype=np.float32).reshape((3, 3))
    # TLC = np.zeros((4,4), dtype=np.float32)
    # TLC[:3, :3] = RLC
    # TLC[3, 3] = 1.0
    qLC = Quaternion(matrix=RLC)
    # qLC = tf.transformations.quaternion_from_matrix(TLC)

    frameID_Aligned = alignedFrameID
    positionAligned = np.zeros((3), dtype=np.float32)

    while ( not rospy.is_shutdown() and count < N ):
        # Frame.
        frameID_Centroid = "PC_%03d" % (count)
        
        f = files[count]
        
        # Read the point cloud. 
        pc = loader(f)

        # Check the dimension. 
        if ( pc.shape[1] < 3 ):
            raise Exception("count = {}, pc.shape = {}. ".format( count, pc.shape ))

        # Extract the vertices.
        vertices = pc[:, :3]

        # Centroid.
        centroid = pcu.centroid_from_vertex_list( vertices )

        # Frame rotation of the point cloud.
        # PCC: Point Cloud Centroid.
        qPCC = Quaternion( axis=(1.0, 0.0, 0.0), radians=-np.pi/2 ) 

        # Convert the point cloud into PointCloud2 message. 
        if ( flagHaveColor ):
            msg = pcu.convert_numpy_2_pointcloud2_color( vertices, frame_id=frameID_Aligned, maxDistColor=colorMaxDist )
        else:
            msg = pcu.convert_numpy_2_pointcloud2(vertices, frame_id=frameID_Aligned)
        
        published = False
        # while ( not rospy.is_shutdown() and not published ):
        while ( not rospy.is_shutdown() and not published ):
            # https://www.theconstructsim.com/ros-qa-158-publish-one-message-topic/
            connections = pub.get_num_connections()
            if ( connections > 0 ):
                # Broadcast tf for the orientation aligned point cloud.
                br.sendTransform(centroid,
                                [ qPCC[1], qPCC[2], qPCC[3], qPCC[0] ],
                                rospy.Time.now(),
                                frameID_Centroid, 
                                "world")
                                
                br.sendTransform(positionAligned,
                                [ qPCC[1], qPCC[2], qPCC[3], qPCC[0] ],
                                rospy.Time.now(),
                                frameID_Aligned, 
                                "world")

                msg.header.stamp = rospy.Time().now()

                pub.publish(msg)
                rospy.loginfo( "pc centroid {}, qPCC {}. ".format(centroid, qPCC) )
                
                # Broadcast tf for the centroid view point. 
                br.sendTransform(centroid,
                                [ qPCC[1], qPCC[2], qPCC[3], qPCC[0] ],
                                rospy.Time.now(),
                                frameID_Centroid, 
                                "world")

                # Broadcast tf for the orientation aligned point cloud.
                br.sendTransform(positionAligned,
                                [ qPCC[1], qPCC[2], qPCC[3], qPCC[0] ],
                                rospy.Time.now(),
                                frameID_Aligned, 
                                "world")

                rate.sleep()
                
                published = True
            else:
                rate.sleep()

        count += 1

    rospy.loginfo("All point clouds published.")

if __name__ == "__main__":
    inDir          = rospy.get_param( "/%s/%s" % (NODE_NAME, "input_dir"     ) )
    pattern        = rospy.get_param( "/%s/%s" % (NODE_NAME, "pattern"       ) )
    ext            = rospy.get_param( "/%s/%s" % (NODE_NAME, "ext"           ) )
    rate           = rospy.get_param( "/%s/%s" % (NODE_NAME, "rate"          ) )
    color          = rospy.get_param( "/%s/%s" % (NODE_NAME, "color"         ) )
    colorMaxDist   = rospy.get_param( "/%s/%s" % (NODE_NAME, "color_max_dist") )
    alignedFrameID = rospy.get_param( "/%s/%s" % (NODE_NAME, "aligned_frame" ), ALIGNED_FRAME_ID)

    # Find all the files.
    files = sorted( glob.glob( "%s/%s%s" % (inDir, pattern, ext) ) )

    if ( 0 == len(files) ):
        raise Exception("No files found at %s with pattern %s and ext %s. " % ( inDir, pattern, ext ))

    if ( colorMaxDist <= 0 ):
        colorMaxDist = None

    if ( ".npy" == ext ):
        loader = pcu.NumPyLoader(flagBinary=True)
    elif ( ".dat" == ext ):
        loader = pcu.NumPyLoader(flagBinary=False)
    elif ( ".ply" == ext ):
        loader = pcu.PLYLoader()
    else:
        raise Exception("Un-recognized ext: %s. " % (ext))

    try:
        # publisher(files, loader, rate, color, colorMaxDist)
        publisher( files, loader, rate, False, colorMaxDist, \
            alignedFrameID=alignedFrameID, \
            nodeName=NODE_NAME, topicName=TOPIC_NAME )
    except rospy.ROSInterruptException:
        pass
