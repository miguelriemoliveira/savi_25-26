#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
from functools import partial
import glob
from random import randint
import cv2  # import the opencv library
# from matplotlib import pyplot as plt
import numpy as np
import argparse
from auxiliary_functions import changeImageColor, objectiveFunction
from scipy.optimize import least_squares

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

    args = vars(parser.parse_args())
    print(args)

    # ----------------------------------
    # Load the query_transformed, query mask and target images stored in this folder
    # ----------------------------------
    q_image = cv2.imread('q_image_transformed.png')
    t_image = cv2.imread('t_image.png')
    q_mask = cv2.imread('q_mask.png')
    # to have a dtype = boll so we can use as mask, e.g. image[q_mask]
    q_mask = (q_mask/255).astype(bool)

    # test my functions

    # q_image_changed = changeImageColor(q_image, s=2.5, b=30, mask=q_mask)  # type: ignore

    # win_name = 'query image original'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(win_name, q_image)  # type: ignore

    # win_name = 'query image changed'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(win_name, q_image_changed)  # type: ignore
    # cv2.waitKey(0)

    # ------------------------------
    # Fusion or stitching
    # ------------------------------
    mosaic_image = deepcopy(t_image)  # the outer part is alreay ok, jsut need to change the middel
    mosaic_image[q_mask] = 0.5 * t_image[q_mask] + 0.5 * q_image[q_mask]
    # mosaic_image[q_mask] = q_image_transformed[q_mask]

    # Convert the mosaic back to unsigned integer 8 bits (uint8)
    mosaic_image = mosaic_image.astype(np.uint8)

    # ---------------------------------------

    win_name = 'target image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, t_image)  # type: ignore

    # win_name = 'query mask'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(win_name, q_mask.astype(np.uint8)*255)  # type: ignore

    # win_name = 'mosaic_image'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(win_name, mosaic_image)  # type: ignore

    # ------------------------------------
    # Start optimization
    # -------------------------------------
    shared_mem = {'q_image': q_image, 't_image': t_image, 'q_mask': q_mask}

    initial_params = [1.0, 0.0]
    result = least_squares(partial(objectiveFunction, shared_mem=shared_mem),
                           initial_params)

    print('Optimization finished. Result=\n' + str(result))

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
