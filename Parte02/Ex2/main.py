#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
import cv2  # import the opencv library
import numpy as np

# main function, where our code should be


def main():
    print("python main function")

    dataset_path = '/home/mike/GoogleDrive/UA/Aulas/2025-2026/1ºSem/SAVI_25-26/savi_25-26/Parte02/cat_dog_savi'

    # Check all image filenames in disk
    image_filenames = glob.glob(dataset_path + "/*.jpeg")
    image_filenames.sort()
    print(image_filenames)

    # Read all images
    images = []  # initialize an empty image list
    for image_filename in image_filenames:
        image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        images.append(image)

    # Shows image
    for image in images:
        cv2.imshow('Image X', image)
        cv2.waitKey(0)

    exit(0)

    # Reading the image from disk
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
