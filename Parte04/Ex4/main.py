#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
from random import randint
import cv2  # import the opencv library
# from matplotlib import pyplot as plt
import numpy as np
import argparse

# import matplotlib
# matplotlib.use('Agg')


def main():

    # ------------------------------------
    # Setu pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser(
        prog='Traffic car couter',
        description='Counts cars',
        epilog='This is finished')

    # parser.add_argument('-qi', '--query_image', type=str, default='../images/santorini/1.png')
    # parser.add_argument('-ti', '--target_image', type=str, default='../images/santorini/2.png')

    parser.add_argument('-qi', '--query_image', type=str, default='../images/castle/1.png')
    parser.add_argument('-ti', '--target_image', type=str, default='../images/castle/2.png')

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
    # q_image_gui = deepcopy(q_image)
    # for key_point in q_key_points:  # iterate all keypoints
    #     x = int(key_point.pt[0])
    #     y = int(key_point.pt[1])

    #     color = (randint(0, 255), randint(0, 255), randint(0, 255))
    #     radius = 30
    #     cv2.circle(q_image_gui, (x, y), radius, color, 3)  # type: ignore

    # # Drawing the keypoints on the target image
    # t_image_gui = deepcopy(t_image)
    # for key_point in t_key_points:  # iterate all keypoints
    #     x = int(key_point.pt[0])
    #     y = int(key_point.pt[1])

    #     color = (randint(0, 255), randint(0, 255), randint(0, 255))
    #     radius = 30
    #     cv2.circle(t_image_gui, (x, y), radius, color, 3)  # type: ignore

    # ----------------------------------
    # Feature matching (with David Lowe strategy)
    # ----------------------------------
    index_params = dict(algorithm=1, trees=15)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    best_matches = flann_matcher.knnMatch(q_descriptors, t_descriptors, k=2)
    # k = 2 so that for every quaery_point we have the best and the second best matches

    q_image_gui = deepcopy(q_image)
    t_image_gui = deepcopy(t_image)
    # Create a list of matches
    matches = []
    for match_idx, match_vector in enumerate(best_matches):

        best_match = match_vector[0]  # to get the cv2.DMatch from the tuple [match = (cv2.DMatch)]

        # matches.append(best_match)  # taking the winnier, period
        second_best_match = match_vector[1]

        if best_match.distance < second_best_match.distance*0.75:
            matches.append(best_match)  # D. Lowe approach
        else:
            continue

        q_idx = best_match.queryIdx
        t_idx = best_match.trainIdx

        # retrieving the coordinates of the points that are associated in this match
        x_q, y_q = int(q_key_points[q_idx].pt[0]), int(q_key_points[q_idx].pt[1])
        x_t, y_t = int(t_key_points[t_idx].pt[0]), int(t_key_points[t_idx].pt[1])

        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(q_image_gui, (x_q, y_q), 30, color, 3)
        cv2.circle(t_image_gui, (x_t, y_t), 30, color, 3)

    # Draw matches using opencv drawmatches
    # image_matches = deepcopy(q_image)
    image_matches = cv2.drawMatches(q_image,  # type: ignore
                                    q_key_points,
                                    t_image,  # type: ignore
                                    t_key_points,
                                    matches, None)  # type: ignore

    win_name = 'query image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, q_image_gui)  # type: ignore
    win_name = 'target image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, t_image_gui)  # type: ignore

    win_name = 'matches_image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, image_matches)  # type: ignore

    cv2.waitKey(0)
    #
    # ------------------------------------
    # Open the video file
    # -------------------------------------


if __name__ == '__main__':
    main()
