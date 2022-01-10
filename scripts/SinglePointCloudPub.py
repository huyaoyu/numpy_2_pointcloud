#!/usr/bin/env python
# -*- coding: future_fstrings -*-

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

# System packages
import glob
import numpy as np
import os
from pyquaternion import Quaternion

# ROS packages.
import rospy
from sensor_msgs.msg import PointCloud2
import tf
import tf2_ros
import geometry_msgs.msg

# Local packages.
import CommonDefinition
import PointCloudUtils as pcu

NODE_NAME  = "pc_static_node"
TOPIC_NAME = "pc_static"
ALIGNED_FRAME_ID = CommonDefinition.COMMON_ALIGNED_FRAME_ID

def get_pc_loader(fn):
    ext = os.path.splitext(fn)[1]

    if ( ".npy" == ext ):
        loader = pcu.NumPyLoader(flagBinary=True)
    elif ( ".dat" == ext ):
        loader = pcu.NumPyLoader(flagBinary=False)
    elif ( ".ply" == ext ):
        loader = pcu.PLYLoader()
    else:
        raise Exception("Un-recognized ext: %s. " % (ext))

    return loader

def broadcast_static_tf(p_frame, c_frame, position, orientation_wxyz):
    br = tf2_ros.StaticTransformBroadcaster()

    static_tf = geometry_msgs.msg.TransformStamped()
    static_tf.header.stamp = rospy.Time.now()
    static_tf.header.frame_id = p_frame
    static_tf.child_frame_id = c_frame
    static_tf.transform.translation.x = position[0]
    static_tf.transform.translation.y = position[1]
    static_tf.transform.translation.z = position[2]
    static_tf.transform.rotation.w = orientation_wxyz[0]
    static_tf.transform.rotation.x = orientation_wxyz[1]
    static_tf.transform.rotation.y = orientation_wxyz[2]
    static_tf.transform.rotation.z = orientation_wxyz[3]

    br.sendTransform(static_tf)

def publisher(file, loader,
        flag_have_color=True,
        color_max_dist=None,
        aligned_frame_id=ALIGNED_FRAME_ID,
        topic_name=TOPIC_NAME):

    position_aligned = np.zeros((3), dtype=np.float32)

    # Frame.
    frame_id_centroid = "PC_000"
    
    # Read the point cloud. 
    pc = loader(file)

    # Check the dimension. 
    if ( pc.shape[1] < 3 ):
        raise Exception(f"pc.shape = {pc.shape}. ")

    # Extract the vertices.
    vertices = pc[:, :3]

    # Centroid.
    centroid = pcu.centroid_from_vertex_list( vertices )

    # Frame rotation of the point cloud.
    # PCC: Point Cloud Centroid.
    qPCC = Quaternion( axis=(1.0, 0.0, 0.0), radians=-np.pi/2 ) 

    # Convert the point cloud into PointCloud2 message. 
    if ( flag_have_color ):
        msg = pcu.convert_numpy_2_pointcloud2_color( vertices, frame_id=aligned_frame_id, maxDistColor=color_max_dist )
    else:
        msg = pcu.convert_numpy_2_pointcloud2(vertices, frame_id=aligned_frame_id)
    
    broadcast_static_tf( 'world', aligned_frame_id, position_aligned, qPCC )
    
    msg.header.stamp = rospy.Time().now()
    pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1, latch=True)
    pub.publish(msg)
    rospy.loginfo( "pc centroid {}, qPCC {}. ".format(centroid, qPCC) )

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    node_name = rospy.get_name()

    print(f'This is {node_name}')

    in_fn            = rospy.get_param( "~%s" % ("in_fn"         ) )
    color            = rospy.get_param( "~%s" % ("color"         ) )
    color_max_dist   = rospy.get_param( "~%s" % ("color_max_dist") )
    aligned_frame_id = rospy.get_param( "~%s" % ("aligned_frame" ), ALIGNED_FRAME_ID)

    if ( color_max_dist <= 0 ):
        color_max_dist = None

    loader = get_pc_loader(in_fn)

    try:
        publisher( in_fn, loader, 
            False, 
            color_max_dist, \
            aligned_frame_id=aligned_frame_id, \
            topic_name=TOPIC_NAME )
    except rospy.ROSInterruptException:
        pass

    print('Begin spinning. ')
    rospy.spin()
