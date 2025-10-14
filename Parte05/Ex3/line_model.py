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
