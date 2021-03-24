import os
import imageio
import argparse
import numpy as np

#############################################################################
# TODO: Add additional imports
#############################################################################

#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


def get_parser():
    parser = argparse.ArgumentParser(description="Points Selection")
    parser.add_argument("image1", type=str, help="path to image 1")
    parser.add_argument("image2", type=str, help="path to image 2")
    return parser


def pick_points(img1, img2):
    """
    Functionality to get manually identified corresponding points from two views.

    Inputs:
    - img1: The first image to select points from
    - img2: The second image to select points from

    Output:
    - coords1: An ndarray of shape (N, 2) with points from image 1
    - coords2: An ndarray of shape (N, 2) with points from image 2
    """
    ############################################################################
    # TODO: Implement pick_points
    ############################################################################

    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################


if __name__ == "__main__":
    args = get_parser().parse_args()

    img1 = np.asarray(imageio.imread(args.image1))
    img2 = np.asarray(imageio.imread(args.image2))

    coords1, coords2 = pick_points(img1, img2)

    assert len(coords1) == len(coords2), "The number of coordinates does not match"

    filename1 = os.path.splitext(args.image1)[0] + ".npy"
    filename2 = os.path.splitext(args.image2)[0] + ".npy"

    assert not os.path.exists(filename1), f"Output file {filename1} already exists"
    assert not os.path.exists(filename2), f"Output file {filename2} already exists"

    np.save(filename1, coords1)
    np.save(filename2, coords2)
