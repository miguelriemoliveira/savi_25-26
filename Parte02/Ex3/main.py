#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
import cv2  # import the opencv library
import numpy as np

# main function, where our code should be


def main():
    print("python main function")

    # --------------------------
    # Read image
    # --------------------------
    image_filename = '/home/mike/GoogleDrive/UA/Aulas/2025-2026/1ºSem/SAVI_25-26/savi_25-26/Parte02/images/praia.png'
    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    h, w, numchannels = image.shape

    cv2.imshow('Beach', image)
    cv2.waitKey(25)  #

    # --------------------------
    # Run segmentation
    # --------------------------
    # We have two classes: sky (255) and "not sky" (0)

    # initialize segmentation mask
    segmentation_mask = np.ndarray((h, w), dtype=np.uint8)

    # iterate all pixels in the image
    for y in range(0, h):  # iterating rows
        for x in range(0, w):  # iterating cols
            b, g, r = image[y, x, :]

            # Segmentation decision
            if b > g and b > r:  # lets assume this is a bluish colored image
                segmentation_mask[y, x] = 255  # this is sky
            else:
                segmentation_mask[y, x] = 0  # this is the not sky

    cv2.imshow('Segmentation mask', segmentation_mask)

    cv2.waitKey(0)  # 0 means wait forever until a key is pressed


if __name__ == '__main__':
    main()
