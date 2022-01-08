
import json
import numpy as np
from pyquaternion import Quaternion

class CameraDescriptor(object):
    def __init__(self):
        super(CameraDescriptor, self).__init__()

        self.id = -1
        self.centroid = None
        self.quaternion = None

    def get_id(self):
        return self.id

    def get_centroid(self):
        return self.centroid

    def get_quaternion(self):
        return self.quaternion

def read_cam_proj_csv(fn):
    pc = np.loadtxt(fn, delimiter=",").astype(np.float32)

    camProjs = []

    for i in range( pc.shape[0] ):
        cd = CameraDescriptor()
        cd.id = int(pc[i, 0])
        cd.centroid = pc[i, 5:8]
        
        q = pc[i, 1:5]
        cd.quaternion = Quaternion( w=q[0], x=q[1], y=q[2], z=q[3] )

        camProjs.append(cd)
    
    return camProjs

def read_cam_proj_json(fn, rootElement="camProjs"):
    camProjs = []
    
    with open(fn, "r") as fp:
        jFp = json.load(fp)

        jCamProjs = jFp["camProjs"]

        for cp in jCamProjs:
            cd = CameraDescriptor()
            cd.id = cp["id"]
            cd.centroid = np.array( cp["T"], dtype=np.float32 )
            q = cp["Q"]
            cd.quaternion = Quaternion( w=q[0], x=q[1], y=q[2], z=q[3] )

            camProjs.append(cd)

        fp.close()
    
    return camProjs