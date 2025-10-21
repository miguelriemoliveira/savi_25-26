#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
from random import randint
import cv2  # import the opencv library
# from matplotlib import pyplot as plt
import numpy as np
import argparse
from auxiliary_functions import changeImageColor

# import matplotlib
# matplotlib.use('Agg')


def main():

    A = np.ndarray((3, 3), dtype=np.uint8)

    print('A=\n' + str(A))

    A_float = A.astype(float)/255

    print('A_float=\n' + str(A_float))

    # Apply the transformation
    s = 1.0
    b = 0.0

    B_float = s*A + b

    print('B_float=\n' + str(B_float))


if __name__ == '__main__':
    main()
