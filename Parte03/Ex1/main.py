#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
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

    parser.add_argument('-if', '--input_filename', type=str, default='../docs/traffic.mp4')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Open the video file
    # -------------------------------------
    capture = cv2.VideoCapture(args['input_filename'])

    # Check if the sequence was opened successfully
    if not capture.isOpened():
        print("Error: Could not open image sequence.")
    else:
        print("Image sequence opened successfully!")

    # ------------------------------------
    # Read and display all frames
    # ------------------------------------
    while True:

        ret, frame = capture.read()
        if ret == False:
            break

        cv2.imshow('Current frame', frame)

        # Break if q is pressed
        key = cv2.waitKey(0)
        # print('key = ' + str(key))
        if key == 113:
            print('You pressed q. Quitting.')
            break

    cv2.destroyAllWindows()  # Close the window


if __name__ == '__main__':
    main()
