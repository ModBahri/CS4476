import numpy as np


def compute_homography(t1, t2):
    """
    Computes the Homography matrix for corresponding image points t1 and t2

    The function should take a list of N ≥ 4 pairs of corresponding points
    from the two views, where each point is specified with its 2d image
    coordinates.

    Inputs:
    - t1: Nx2 matrix of image points from view 1
    - t2: Nx2 matrix of image points from view 2

    Returns a tuple of:
    - H: 3x3 Homography matrix
    """
    H = np.eye(3)
    #############################################################################
    # TODO: Compute the Homography matrix H for corresponding image points t1, t2
    #############################################################################
    L = np.empty((t1.shape[0] * 2, 9))
    i = 0
    j = 0
    while j < t1.shape[0]:
        pt = np.concatenate((t1[j], [1]))
        L[i] = np.concatenate((pt, [0, 0, 0], np.multiply(pt, -t2[j][0])))
        L[i + 1] = np.concatenate(([0, 0, 0], pt, np.multiply(pt, -t2[j][1])))
        i += 2
        j += 1
    lt = np.transpose(L)
    ltl = np.matmul(lt, L)
    val, vect = np.linalg.eig(ltl)
    vect = np.transpose(vect)
    H = vect[np.argmin(val)].reshape((3, 3))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return H