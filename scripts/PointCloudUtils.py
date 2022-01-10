
import numpy as np
from plyfile import PlyData, PlyElement
from pyquaternion import Quaternion

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import re, string
pattern = re.compile("[\W_]+")

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

def centroid_from_vertex_list(vertices):
    if ( vertices.size == 0 ):
        raise Exception("Zero vertex in the list. Cannot compute the centroid. ")

    avgX = vertices[:,0].mean()
    avgY = vertices[:,1].mean()
    avgZ = vertices[:,2].mean()

    return np.array( [avgX, avgY, avgZ], dtype=np.float32 )

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

    ROS sensor_msgs/PointField.
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

    # Structured array.
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

class PCLoader(object):
    def __init__(self):
        super(PCLoader, self).__init__()

    def __call__(self, fn):
        raise Exception("Base class __call__ has been called.")

class NumPyLoader(PCLoader):
    def __init__(self, flagBinary=True, dtype=np.float32, delimiter=" "):
        super(NumPyLoader, self).__init__()

        self.flagBinary = flagBinary
        self.dtype = dtype
        self.delimiter = delimiter

    def __call__(self, fn):
        if ( self.flagBinary ):
            return np.load(fn).astype(self.dtype)
        else:
            return np.loadtxt(fn, delimiter=self.delimiter).astype(self.dtype)

class PLYLoader(PCLoader):
    def __init__(self, dtype=np.float32):
        super(PLYLoader, self).__init__()

        self.dtype = dtype

    def __call__(self, fn):
        ply = PlyData.read(fn)

        # Check if we have the normal fields.
        flag_have_normal = True
        try:
            ply["vertex"].ply_property("nx")
        except KeyError as ex:
            print("nx" == pattern.sub( "", str(ex)) )
            print("No normal fields in the PLY file %s. " % (fn) )
            flag_have_normal = False

        flagHaveCurvature = True
        try:
            ply["vertex"].ply_property("curvature")
        except KeyError as ex:
            flagHaveCurvature = False

        if ( flag_have_normal ):
            if ( flagHaveCurvature ):
                array = np.vstack( ( ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"], \
                    ply["vertex"]["nx"], ply["vertex"]["ny"], ply["vertex"]["nz"], \
                    ply["vertex"]["curvature"] ) ).astype(self.dtype)
            else:
                array = np.vstack( ( ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"], \
                    ply["vertex"]["nx"], ply["vertex"]["ny"], ply["vertex"]["nz"] ) 
                        ).astype(self.dtype)
        else:
            array = np.vstack( 
                ( ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"] ) 
                    ).astype(self.dtype)

        return array.transpose()