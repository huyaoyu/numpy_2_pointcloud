#!/usr/bin/env python

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

import glob
import numpy as np
import os
from pyquaternion import Quaternion

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf

from ColorMapping import color_map

DIST_COLORS = [\
    "#2980b9",\
    "#27ae60",\
    "#f39c12",\
    "#c0392b",\
    ]

DIST_COLOR_LEVELS = 20

def from_quaternion_to_rotation_matrix(q):
    """
    q: A numpy vector, 4x1.
    """

    qi2 = q[0, 0]**2
    qj2 = q[1, 0]**2
    qk2 = q[2, 0]**2

    qij = q[0, 0] * q[1, 0]
    qjk = q[1, 0] * q[2, 0]
    qki = q[2, 0] * q[0, 0]

    qri = q[3, 0] * q[0, 0]
    qrj = q[3, 0] * q[1, 0]
    qrk = q[3, 0] * q[2, 0]

    s = 1.0 / ( q[3, 0]**2 + qi2 + qj2 + qk2 )
    ss = 2 * s

    R = [\
        [ 1.0 - ss * (qj2 + qk2), ss * (qij - qrk), ss * (qki + qrj) ],\
        [ ss * (qij + qrk), 1.0 - ss * (qi2 + qk2), ss * (qjk - qri) ],\
        [ ss * (qki - qrj), ss * (qjk + qri), 1.0 - ss * (qi2 + qj2) ],\
    ]

    R = np.array(R, dtype=np.float32)

    return R

def get_pose_from_line(poseDataLine):
    """
    poseDataLine is a 7-element NumPy array. The first 3 elements are 
    the translations. The remaining 4 elements are the orientation 
    represented as a quternion.
    """

    data = poseDataLine.reshape((-1, 1))
    t = data[:3, 0].reshape((-1, 1))
    q = data[3:, 0].reshape((-1, 1))
    R = from_quaternion_to_rotation_matrix(q)

    return R.transpose(), -R.transpose().dot(t), q

def convert_numpy_2_pointcloud2(points, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points. 

    points: A NumPy array of Nx3.
    stamp: An alternative stamp.
    frame_id: The frame id. String.

    This function is a combinateion of 
    the code at https://github.com/spillai/pybot/blob/master/pybot/externals/ros/pointclouds.py
    and expo_utility.xyz_array_to_point_cloud_msg() function of the AirSim package.
    '''
    
    header = Header()

    if stamp is None:
        header.stamp = rospy.Time().now()
    else:
        header.stamp = stamp

    if frame_id is None:
        header.frame_id = "None"
    else:
        header.frame_id = frame_id

    msg = PointCloud2()
    msg.header = header

    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width  = points.shape[0]

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]

    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = msg.point_step * points.shape[0]
    msg.is_dense     = int( np.isfinite(points).all() )
    msg.data         = np.asarray(points, np.float32).tostring()

    return msg

def convert_numpy_2_pointcloud2_color(points, stamp=None, frame_id=None, maxDistColor=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points. 

    This function will automatically assign RGB values to each point. The RGB values are
    determined by the distance of a point from the origin. Use maxDistColor to set the distance 
    at which the color corresponds to the farthest distance is used.

    points: A NumPy array of Nx3.
    stamp: An alternative ROS header stamp.
    frame_id: The frame id. String.
    maxDisColor: Should be positive if specified..

    This function get inspired by 
    https://github.com/spillai/pybot/blob/master/pybot/externals/ros/pointclouds.py
    https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
    (https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/)
    and expo_utility.xyz_array_to_point_cloud_msg() function of the AirSim package.

    ROS sensor_msgs/PointField Message.
    http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html

    More references on mixed-type NumPy array, structured array.
    https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    https://stackoverflow.com/questions/37791134/merge-width-x-height-x-3-numpy-uint8-array-into-width-x-height-x-1-uint32-array
    https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
    '''
    
    # Clipping input.
    dist = np.linalg.norm( points, axis=1 )
    if ( maxDistColor is not None and maxDistColor > 0):
        dist = np.clip(dist, 0, maxDistColor)

    # Compose color.
    cr, cg, cb = color_map( dist, DIST_COLORS, DIST_COLOR_LEVELS )

    C = np.zeros((cr.size, 4), dtype=np.uint8) + 255

    C[:, 0] = cb.astype(np.uint8)
    C[:, 1] = cg.astype(np.uint8)
    C[:, 2] = cr.astype(np.uint8)

    C = C.view("uint32")

    # Concatenate.
    pointsColor = np.zeros( (points.shape[0], 1), \
        dtype={ 
            "names": ( "x", "y", "z", "rgba" ), 
            "formats": ( "f4", "f4", "f4", "u4" )} )

    points = points.astype(np.float32)

    pointsColor["x"] = points[:, 0].reshape((-1, 1))
    pointsColor["y"] = points[:, 1].reshape((-1, 1))
    pointsColor["z"] = points[:, 2].reshape((-1, 1))
    pointsColor["rgba"] = C

    header = Header()

    if stamp is None:
        header.stamp = rospy.Time().now()
    else:
        header.stamp = stamp

    if frame_id is None:
        header.frame_id = "None"
    else:
        header.frame_id = frame_id

    msg = PointCloud2()
    msg.header = header

    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width  = points.shape[0]

    msg.fields = [
        PointField('x',  0, PointField.FLOAT32, 1),
        PointField('y',  4, PointField.FLOAT32, 1),
        PointField('z',  8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
        ]

    msg.is_bigendian = False
    msg.point_step   = 16
    msg.row_step     = msg.point_step * points.shape[0]
    msg.is_dense     = int( np.isfinite(points).all() )
    msg.data         = pointsColor.tostring()

    return msg

def publisher(files, poses, r=1, flagHaveColor=True, colorMaxDist=None):
    pub = rospy.Publisher("sim_pc", PointCloud2, queue_size=1)
    br  = tf.TransformBroadcaster()

    rospy.init_node("sim_pc_publisher", anonymous=True)
    rate = rospy.Rate(r)

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

    while ( not rospy.is_shutdown() and count < N ):
        f = files[count]
        
        # Read the point cloud. 
        pc = np.load(f).astype(np.float32)

        # Get the pose. 
        pose = poses[count, :]

        # R, t, q = get_pose_from_line(pose)
        # RR = np.zeros((4,4), dtype=np.float32)
        # RR[:3,:3] = R
        # RR[3,3]   = 1.0

        t = pose[:3]
        q = pose[3:]

        qCA = Quaternion( q[3], q[0], q[1], q[2])

        qLW = qAW * qCA * qLC

        # qLW = tf.transformations.quaternion_multiply( q, qLC )

        # Broadcast tf. 
        br.sendTransform(qAW.rotation_matrix.dot(t),
                     [ qLW.elements[1], qLW.elements[2], qLW.elements[3], qLW.elements[0] ],
                     rospy.Time.now(),
                     "LIDAR",
                     "world")

        # [ qLW.elements[1], qLW.elements[2], qLW.elements[3], qLW.elements[0] ]

        # Check the dimension. 
        if ( pc.shape[1] != 3 ):
            raise Exception("count = {}, pc.shape = {}. ".format( count, pc.shape ))

        # Convert the point cloud into PointCloud2 message. 
        if ( flagHaveColor ):
            msg = convert_numpy_2_pointcloud2_color( pc, frame_id="LIDAR", maxDistColor=colorMaxDist )
        else:
            msg = convert_numpy_2_pointcloud2(pc, frame_id="LIDAR")
        
        pub.publish(msg)
        rospy.loginfo("pc %s published. " % ( f ))

        rate.sleep()

        count += 1

    rospy.loginfo("All point clouds published.")

if __name__ == "__main__":
    inDir    = rospy.get_param("/sim_pc_publisher/input_dir")
    poseFile = rospy.get_param("/sim_pc_publisher/pose_file")
    suffix   = rospy.get_param("/sim_pc_publisher/suffix")
    ext      = rospy.get_param("/sim_pc_publisher/ext")
    rate     = rospy.get_param("/sim_pc_publisher/rate")
    color    = rospy.get_param("/sim_pc_publisher/color")
    colorMaxDist = rospy.get_param("/sim_pc_publisher/color_max_dist")

    # Find all the files.
    files = sorted( glob.glob( "%s/*%s%s" % (inDir, suffix, ext) ) )

    if ( 0 == len(files) ):
        raise Exception("No files found at %s with suffix %s and ext %s. " % ( inDir, suffix, ext ))

    # Load the poses.
    poses = np.loadtxt( "%s/%s" % (inDir, poseFile), dtype=np.float32 )

    if ( poses.shape[1] != 7 ):
        raise Exception("poses has wrong shape. poses.shape = {}. ".format( poses.shape ))

    if ( poses.shape[0] != len(files) ):
        raise Exception("poses and files have different lengths. pose.shape[0] = %d, len(files) = %d. " % \
            ( poses.shape[0], len(files) ) )

    if ( colorMaxDist <= 0 ):
        colorMaxDist = None

    try:
        publisher(files, poses, rate, color, colorMaxDist)
    except rospy.ROSInterruptException:
        pass
