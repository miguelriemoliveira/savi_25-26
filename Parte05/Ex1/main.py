#!/usr/bin/env python3
# shebang line for linux / mac


import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

import matplotlib
matplotlib.use('QtAgg')
# Select one option that works for your OS
# https://matplotlib.org/stable/users/explain/figure/backends.html


def main():

    # ------------------------------------
    # Setup pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser(prog='Optimization for a line')

    parser.add_argument('-f', '--filename', type=str, default='./data.json')
    parser.add_argument(
        '-n', '--npoints', type=int, default=10,
        help='Number of points clicked. You can also use ESC if you want to stop before that.')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Setup matplotlib
    # ------------------------------------
    plt.axis([-10, 10, -5, 5])

    points = plt.ginput(n=args['npoints'])  # n is the number of clicks (and points)
    print('result = ' + str(points))

    # Creatign the lists of xs and ys
    xs = []  # list of x coords for all the points
    ys = []
    for point in points:
        x = float(point[0])
        y = float(point[1])
        xs.append(x)
        ys.append(y)

    print('xs=' + str(xs))
    print('ys=' + str(ys))

    # Create a  dictionary with the xs and ys coords
    d = {'xs': xs, 'ys': ys}

    with open(args['filename'], "w") as f:
        f.write(json.dumps(d, indent=0))

    print('Saved file ' + args['filename'])

    # plt.show()  # wait for user to destriy window
    # plt.waitforbuttonpress(0.1)


if __name__ == '__main__':
    main()
