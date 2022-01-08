import numpy as np

def find_quaternion_between_vectors( v0, v1 ):
    """
    v0, v1 are numpy arrays.
    """

    a = np.cross( v0, v1 )

    n0 = np.linalg.norm( v0, 2 )
    n1 = np.linalg.norm( v1, 2 )

    w = np.sqrt( n0**2 * n1**2 ) + v0.transpose().dot( v1 )

    q = np.array( ( a[0], a[1], a[2], w ), dtype=np.float32 )
    q = q / np.linalg.norm(q,2)

    return q