#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import cv2  # import the opencv library
import numpy as np

# main function, where our code should be


def main():
    print("python main function")

    # Reading the image from disk
    original_image = cv2.imread('lake.jpg', cv2.IMREAD_COLOR)
    # cv2.imshow('Original', image)

    # Make the image darker on the right hand side
    h, w, channels = original_image.shape
    factor = 1
    middle_width = round(w / 2)

    factors = np.linspace(1, 0, 100)
    print('factors = ' + str(factors))

    for factor in factors:  # progressive nightfall

        # createa  new copy of the original to make sure we start from the beggining
        image = deepcopy(original_image)

        for y in range(0, h):  # iterate all rows
            for x in range(middle_width, w):  # iterating cols from middle to last
                bgr = original_image[y, x, :]  # get the value from the original image
                bgr_darkened = bgr * factor
                image[y, x, :] = bgr_darkened  # daken the imagw

        cv2.imshow('Darkened', image)
        cv2.waitKey(25)  #

    cv2.waitKey(0)  # 0 means wait forever until a key is pressed


if __name__ == '__main__':
    main()
