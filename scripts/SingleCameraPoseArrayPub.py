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

NODE_NAME  = "pc_camera_pose_node"
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

class CameraPosePublisher(object):
    def __init__(self, 
        topic_name, 
        camera_arrow_color=(0.0, 0.0, 1.0, 0.75),
        camera_arrow_scale=(0.2, 0.02, 0.02),
        camera_text_color=(0.0, 1.0, 0.0, 0.75),
        camera_text_scale=0.05,
        trajectory_color=(1.0, 0.0, 0.0, 0.75),
        trajectory_scale=0.02,
        ):
        super(CameraPosePublisher, self).__init__()

        self.topic_name = topic_name
        self.topic_name_traj = '%s_traj' % (self.topic_name)
        self.camera_arrow_color = ColorRGBA(*camera_arrow_color)
        self.camera_arrow_scale = Vector3(*camera_arrow_scale)
        self.camera_text_color  = ColorRGBA(*camera_text_color)
        self.camera_text_scale  = Vector3(0, 0, camera_text_scale)
        self.trajectory_color   = ColorRGBA(*trajectory_color)
        self.trajectory_scale   = Vector3(trajectory_scale, 0, 0)

        self.q_x2z = Quaternion( axis=(0,1,0), radians=-np.pi/2 )
        self.cam_des = None # A list of camera description.

    def read_camera_descriptions(self, fn):
        self.cam_des = read_cameras_as_camera_descriptors(fn)
        rospy.loginfo("%d cameras loaded. " % ( len(self.cam_des) ) )

    def publish(self, aligned_frame=ALIGNED_FRAME_ID):
        pub = rospy.Publisher(self.topic_name, MarkerArray, queue_size=1, latch=True)
        pub_traj = rospy.Publisher(self.topic_name_traj, Marker, queue_size=1, latch=True)

        countMarker_camera = 0
        countMarker_text   = 0

        assert( self.cam_des is not None )

        markers = []

        # The line strip
        marker_trajectory = Marker( \
            header=Header(frame_id=aligned_frame),
            ns="MarkerCameraPoseTraj",
            id=0,
            type=Marker.LINE_STRIP,
            action=0,
            pose=Pose(Point(0, 0, 0), Quaternion(1, 0, 0, 0)),
            scale=Vector3(0.02, 0, 0),
            color=self.trajectory_color )

        for idx in range( len(self.cam_des) ):
            if ( rospy.is_shutdown() ):
                break

            # Centroid.
            centroid = self.cam_des[idx].get_centroid()

            # Quaternion
            qC = self.cam_des[idx].get_quaternion()

            # Combined quaternion.
            qZ = qC * self.q_x2z

            # Pose.
            p = Pose()
            p.position.x    = centroid[0]
            p.position.y    = centroid[1]
            p.position.z    = centroid[2]
            p.orientation.w = qZ[0] # This is w.
            p.orientation.x = qZ[1]
            p.orientation.y = qZ[2]
            p.orientation.z = qZ[3]

            # Save to the line strip point list.
            marker_trajectory.points.append( p.position )

            # The marker.
            markerArrow = Marker( \
                header=Header(frame_id=aligned_frame),
                ns="MarkerCameraPose",
                id=countMarker_camera,
                type=Marker.ARROW,
                action=0,
                pose=p,
                scale=self.camera_arrow_scale,
                color=self.camera_arrow_color )

            countMarker_camera += 1

            markerText = Marker( \
                header=Header(frame_id=aligned_frame),
                ns="MarkerCameraPoseName",
                id=countMarker_text,
                type=Marker.TEXT_VIEW_FACING,
                action=0,
                pose=p,
                scale=self.camera_text_scale,
                color=self.camera_text_color,
                text="C%04d" % (self.cam_des[idx].get_id()) )

            countMarker_text += 1

            # lifetime=rospy.Duration(0)
            
            markers.append(markerArrow)
            markers.append(markerText)

        markerArray = MarkerArray(markers=markers)

        if (not rospy.is_shutdown()):
            pub.publish(markerArray)
            pub_traj.publish(marker_trajectory)
            rospy.loginfo("MarkerArray published with %d markers." % (len(markers)))

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    node_name = rospy.get_name()
    
    in_fn         = rospy.get_param( "~%s" % ("in_fn"         ) )
    aligned_frame = rospy.get_param( "~%s" % ("aligned_frame" ), ALIGNED_FRAME_ID)
    arrow_scale   = rospy.get_param( "~%s" % ("arrow_scale"   ))
    text_scale    = rospy.get_param( "~%s" % ("text_scale"    ))
    traj_scale    = rospy.get_param( "~%s" % ("traj_scale"    ))

    publisher = CameraPosePublisher( TOPIC_NAME, 
        camera_arrow_scale=( arrow_scale, arrow_scale*0.1, arrow_scale*0.1 ),
        camera_text_scale=text_scale, 
        trajectory_scale=traj_scale)
    publisher.read_camera_descriptions(in_fn)
    publisher.publish(aligned_frame=aligned_frame)

    rospy.spin()
