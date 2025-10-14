#!/usr/bin/env python3
# shebang line for linux / mac


import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

import matplotlib
matplotlib.use('QtAgg')
# Select one option that works for your OS
# https://matplotlib.org/stable/users/explain/figure/backends.html


class LineModel:

    def __init__(self, m=None, b=None):

        self.m = m
        self.b = b
        self.plot_handle = None

        # Randomize if m and b are not defined
        if self.m is None or self.b is None:
            self.randomizeParameters()

        print('Created a line model:\n' + self.__str__())

    def randomizeParameters(self):
        self.m = random.uniform(-2, 2)
        self.b = random.uniform(-10, 10)

    def draw(self, color='b'):

        # linspace_example = np.linspace(-10, 10, 100)

        xs = [-10, 10]
        ys = self.getYs(xs)

        if self.plot_handle is None:
            # plot the first time and get the drawing handle
            self.plot_handle = plt.plot(xs, ys, '-' + color, linewidth=2)
        else:
            plt.setp(self.plot_handle, xdata=xs, ydata=ys)

    def getYs(self, xs):
        ys = []
        for x in xs:
            y = self.m * x + self.b  # apply the line equation
            ys.append(y)  # appens to the list

        return ys

    def getError(self, xs_gt, ys_gt):

        ys = self.getYs(xs_gt)  # usign my model, compute the y coordinates for the gt_xs

        # Computing the errors
        errors = []
        for y, y_gt in zip(ys, ys_gt):
            error = abs(y_gt - y)
            errors.append(error)

        # Compute the average error
        n = len(xs_gt)
        total = 0
        for error in errors:
            total += error

        average_error = total / n
        return average_error

    def __str__(self):
        return "Line m=" + str(self.m) + ' b=' + str(self.b)

# How to use the class
# it will set the values of self.m and self.b from the values of m and b
# line2 = LineModel(m=3, b=5)  # named arguments -> define the name order is not importante
# line2 = LineModel(b=5, m=3)  # named arguments -> define the name order is not importante
# unnamed arguments -> will use the order oif th arguments to atribute values
# line2 = LineModel(3, 5)
# line3 = LineModel(b=3)


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

    plt.draw()  # draw in a non blocking fashion

    minimum_error = None
    best_line = LineModel()

    for i in range(0, args['number_iterations']):

        error = line_model.getError(xs_gt=xs_gt, ys_gt=ys_gt)
        print('Line ' + str(i) + ' error = ' + str(error))

        # Check if this current line is the best line
        if minimum_error is None or error < minimum_error:
            # store the parameters of the current line as the best line
            best_line.m = line_model.m
            best_line.b = line_model.b
            minimum_error = error

        line_model.draw()  # draw the line
        best_line.draw(color='r')  # draw the best line in red
        plt.draw()  # draw in a non blocking fashion
        plt.waitforbuttonpress(0.1)  # wait for a second

        line_model.randomizeParameters()  # Change the line's parameters

    plt.show()  # keep the window open when the program finishes


if __name__ == '__main__':
    main()
