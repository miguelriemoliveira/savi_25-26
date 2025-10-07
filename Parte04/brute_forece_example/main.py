#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
from random import randint
import cv2  # import the opencv library
from matplotlib import pyplot as plt
import numpy as np
import argparse

import matplotlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

matplotlib.use('gtk3agg')


def main():

    # ------------------------------------
    # Setu pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser(
        prog='Traffic car couter',
        description='Counts cars',
        epilog='This is finished')

    parser.add_argument('-qi', '--query_image', type=str, default='../images/santorini/1.png')
    parser.add_argument('-ti', '--target_image', type=str, default='../images/santorini/1.png')

    args = vars(parser.parse_args())
    print(args)

    # ----------------------------------
    # Load the query and target images
    # ----------------------------------
    img1 = cv.imread(args['query_image'], cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread(args['target_image'], cv.IMREAD_GRAYSCALE)  # trainImage

    # Initiate SIFT detector
    sift = cv.SIFT_create(nfeatures=500)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3), plt.show()

#
# ------------------------------------
# Open the video file
# -------------------------------------


if __name__ == '__main__':
    main()
