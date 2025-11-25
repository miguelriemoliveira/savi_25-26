#!/usr/bin/env python3
# shebang line for linux / mac

import glob
from random import randint
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse
from dataset import Dataset

from torchvision import transforms


def main():

    # ------------------------------------
    # Setu pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('-df', '--dataset_folder', type=str,
                        default='/home/mike/data/savi_datasets/mnist')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Create datasets
    # ------------------------------------
    dataset = Dataset(args, is_train=True)

    # ------------------------------------
    # Ex1 c)
    # ------------------------------------
    # call getitem for an idx and print the resutl
    image_tensor, label_tensor = dataset.__getitem__(87)  # type: ignore

    print('Image tensor shape: ' + str(image_tensor.shape))
    print('Label tensor: ' + str(label_tensor))

    # Display the image
    to_pil = transforms.ToPILImage()
    image = to_pil(image_tensor)  # get the image from the tensor

    plt.figure()
    plt.imshow(image, cmap='gray')

    # Get teh value of the digit to put on the title
    label = label_tensor.tolist()  # get the value from the tensor
    # print('Label = ' + str(label))
    # max_value = max(label)
    # print('Max value = ' + str(max_value))
    # max_index = label.index(max_value)
    # print('Max index = ' + str(max_index))

    label_name = label.index(max(label))
    plt.title('Label ' + str(label_name))

    plt.axis("off")
    plt.show()

    # ------------------------------------
    # Ex1 d)
    # ------------------------------------
    # Show a mosaic of images
    # Homework


if __name__ == '__main__':
    main()
