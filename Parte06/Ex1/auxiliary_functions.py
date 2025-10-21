#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
from random import randint
import random
import cv2  # import the opencv library
# from matplotlib import pyplot as plt
import numpy as np
import argparse

# import matplotlib
# matplotlib.use('Agg')


def changeImageColor(image_in, s, b, mask=None):

    # image_in is uint8 -> [0 (black), 255 (white)]
    # [0, 255] * 1.0 -> float image

    # Convert the image to float type
    image_in_float = image_in.astype(float)  # 0 = black, 1=white

    image_out_float = image_in_float * s + b  # apply model

    image_out_float[image_out_float > 255] = 255  # saturate to 255
    image_out_float[image_out_float < 0] = 0  # undersaturate to 0

    # b = 0 255

    # convert back to uint8
    image_out = (image_out_float).astype(np.uint8)

    # Copy back the original values of image_in in the regions where the mask is zero
    if mask is not None:
        mask_negated = np.logical_not(mask)
        image_out[mask_negated] = image_in[mask_negated]

    return image_out


# Define the objective function
def objectiveFunction(params, shared_mem):
    # minimuzed version = objectiveFunction(params), shared_mem was already given

    # Extract the parameters
    # params = [s , b]
    s = params[0]
    b = params[1]
    print('s = ' + str(s))
    print('b = ' + str(b))
    q_image = shared_mem['q_image']

    # Applying the image model
    q_image_out = changeImageColor(q_image, s=s, b=b, mask=shared_mem['q_mask'])

    # Compute the error
    # TODO how can we do this?
    error = random.random()

    # TODO recompute the mosaic and show it

    # Draw the new line
    win_name = 'query image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, q_image_out)  # type: ignore
    cv2.waitKey(50)

    print('error = ' + str(error))
    return error  # can be a scalar or a list
