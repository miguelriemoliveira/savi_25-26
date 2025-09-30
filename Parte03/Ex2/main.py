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

    # Define bbox coordinates
    bbox = {'x': 699, 'y': 300, 'w': 808-699, 'h': 402-300}
    # Copute bottom right corner
    bbox['x2'] = bbox['x'] + bbox['w']
    bbox['y2'] = bbox['y'] + bbox['h']

    # ------------------------------------
    # Read and display all frames
    # ------------------------------------
    previous_average = None
    previous_change_event = False
    number_of_cars = 0
    frame_count = 0
    while True:

        ret, frame = capture.read()
        if ret == False:
            break

        # create image for drawing
        frame_gui = deepcopy(frame)

        # Draw the frame count
        cv2.putText(
            frame_gui, '#frame ' + str(frame_count),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the box on the image
        cv2.rectangle(frame_gui, (bbox['x'], bbox['y']), (bbox['x2'], bbox['y2']),
                      (255, 0, 0), 2)

        # Compute grayscale image for analysing
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        box_values = frame_gray[bbox['y']:bbox['y2'],  bbox['x']:bbox['x2']]
        print('box_values = ' + str(box_values))

        average = round(np.mean(box_values), 1)  # type: ignore
        # print('average = ' + str(average))
        cv2.putText(
            frame_gui, 'mean ' + str(average),
            (bbox['x'],
             bbox['y'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Compute the difference between the previous and the current
        if previous_average is None:
            difference = 0  # TODO João has questions
        else:
            difference = round(abs(average - previous_average), 1)

        previous_average = average  # update previous average
        cv2.putText(
            frame_gui, 'dif ' + str(difference),
            (bbox['x'],
             bbox['y'] - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Change detection event
        change_detection_threshold = 10
        if difference > change_detection_threshold:
            change_event = True
        else:
            change_event = False

        # Selecting color for putting the change event text as a function of the value
        if change_event == True:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.putText(
            frame_gui, 'event ' + str(change_event),
            (bbox['x'],
             bbox['y'] - 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # ---------------------------
        # Count a car
        # ---------------------------
        # Option 1: Count the rising edges
        if previous_change_event == False and change_event == True:  # a rising edge
            number_of_cars += 1

        previous_change_event = change_event

        # Draw the number of cars
        cv2.putText(
            frame_gui, '#cars ' + str(number_of_cars),
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw image
        cv2.imshow('Image GUI', frame_gui)
        # cv2.imshow('Image gray', frame_gray)

        # Break if q is pressed
        key = cv2.waitKey(0)
        # print('key = ' + str(key))
        if key == 113:
            print('You pressed q. Quitting.')
            break

        frame_count += 1  # update the frame count

    cv2.destroyAllWindows()  # Close the window


if __name__ == '__main__':
    main()
