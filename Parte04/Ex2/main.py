#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
from random import randint
import cv2  # import the opencv library
import numpy as np
import argparse


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
    q_image = cv2.imread(args['query_image'])
    q_gray_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)  # type: ignore

    t_image = cv2.imread(args['target_image'])
    t_gray_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)  # type: ignore

    # ----------------------------------
    # Compute sift keypoints and descriptors
    # ----------------------------------
    sift = cv2.SIFT_create(nfeatures=500)  # create an object for the sift
    q_key_points, q_descriptors = sift.detectAndCompute(q_gray_image, None)
    t_key_points, t_descriptors = sift.detectAndCompute(t_gray_image, None)

    # Drawing the keypoints on the query image
    q_image_gui = deepcopy(q_image)
    for key_point in q_key_points:  # iterate all keypoints
        x = int(key_point.pt[0])
        y = int(key_point.pt[1])

        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        radius = 30
        cv2.circle(q_image_gui, (x, y), radius, color, 3)  # type: ignore

    # Drawing the keypoints on the target image
    t_image_gui = deepcopy(t_image)
    for key_point in t_key_points:  # iterate all keypoints
        x = int(key_point.pt[0])
        y = int(key_point.pt[1])

        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        radius = 30
        cv2.circle(t_image_gui, (x, y), radius, color, 3)  # type: ignore

    # gui_image = cv2.drawKeypoints(gray_image, key_points, image)

    win_name = 'query image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, q_image_gui)  # type: ignore

    win_name = 'target image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, t_image_gui)  # type: ignore

    cv2.waitKey(0)

    # ------------------------------------
    # Open the video file
    # -------------------------------------


if __name__ == '__main__':
    main()
