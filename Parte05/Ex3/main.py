#!/usr/bin/env python3
# shebang line for linux / mac


import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from scipy.optimize import least_squares
from line_model import LineModel

import matplotlib
matplotlib.use('QtAgg')
# Select one option that works for your OS
# https://matplotlib.org/stable/users/explain/figure/backends.html


def main():

    # ------------------------------------
    # Setup pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser(prog='Optimization for a line')

    parser.add_argument('-f', '--filename', type=str, default='../Ex1/data.json')
    parser.add_argument('-ni', '--number_iterations', type=int, default=500)

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Setup matplotlib
    # ------------------------------------
    plt.axis([-10, 10, -10, 10])
    plt.grid()

    # ------------------------------------
    # Load points from the json and show them in the the image
    # ------------------------------------
    with open(args['filename'], 'r') as file:
        data = json.load(file)

    # print('data = ' + str(data))
    # print(json.dumps(data, indent=2))

    xs_gt = data['xs']
    ys_gt = data['ys']
    plt.plot(xs_gt, ys_gt, '.k', markersize=12)

    # ------------------------------------
    # Create the line model
    # ------------------------------------
    line_model = LineModel()  # it will randomize m and b params

    # ------------------------------------
    # Optimization
    # ------------------------------------

    # Define the objective function
    def objectiveFunction(params):

        # Extract the parameters to class properties
        # We will define a convention in which
        # params = [m , b]
        line_model.m = params[0]
        line_model.b = params[1]

        # Compute the error
        error = line_model.getError(xs_gt=xs_gt, ys_gt=ys_gt)
        print('error = ' + str(error))

        # Draw the new line
        line_model.draw()
        plt.draw()  # draw in a non blocking fashion
        plt.waitforbuttonpress(0.1)  # wait for a second

        return error  # can be a scalar or a list

    # start optimization
    result = least_squares(objectiveFunction, [line_model.m, line_model.b])

    print('Raw result of leas squares = ' + str(result))

    print('Solution: m=' + str(result.x[0]) + ' b=' + str(result.x[1]))
    plt.show()  # keep the window open when the program finishes


if __name__ == '__main__':
    main()

y = a(x-h) ^ 2 + k
