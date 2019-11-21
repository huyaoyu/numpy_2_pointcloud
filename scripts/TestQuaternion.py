from pyquaternion import Quaternion

import numpy as np

if __name__ == "__main__":
    R0 = np.array([ 1, 0, 0, 0, -1, 0, 0, 0, -1 ], dtype=np.float32).reshape((3, 3))
    R1 = np.array([ 0, -1, 0, 1, 0, 0, 0, 0, 1 ], dtype=np.float32).reshape((3, 3))

    q0 = Quaternion(matrix=R0)
    q1 = Quaternion(matrix=R1)

    R = R1.dot(R0)
    q = q1 * q0

    print(R)
    print(q.rotation_matrix)

    print(type(q.rotation_matrix))


