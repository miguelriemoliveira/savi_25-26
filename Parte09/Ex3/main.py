#!/usr/bin/env python3
# shebang line for linux / mac

import glob
from random import randint
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse

import torch
from dataset import Dataset

from torchvision import transforms
from model import Model
from trainer import Trainer


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
    train_dataset = Dataset(args, is_train=True)
    test_dataset = Dataset(args, is_train=False)

    # ------------------------------------
    # Create the model
    # ------------------------------------
    model = Model()

    # ------------------------------------
    # Start trainin
    # ------------------------------------
    trainer = Trainer(args, train_dataset, test_dataset, model)

    trainer.train()  # run training

    # ------------------------------------
    # Ex2 c)
    # ------------------------------------
    # The goal is to call the network forward method with an example from getitem
    # and then get the digit predicted by the network

    # call getitem for an idx and print the resutl
#     image_tensor, label_gt_tensor = dataset.__getitem__(107)  # type: ignore

#     label_pred_tensor = model.forward(image_tensor)

#     print('label_gt_tensor: ' + str(label_gt_tensor))
#     print('label_pred_tensor: ' + str(label_pred_tensor))
#     print('label_pred_tensor sum: ' + str(label_pred_tensor.sum(dim=1)))

#     # label_pred_tensor is the raw output of the network (logits). Its not in the format of probabilities yet.
#     # To convert to probabilities we can use the softmax function

#     label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)
#     print('label_pred_probabilities_tensor: ' +
#           str(label_pred_probabilities_tensor))
#     print('label_pred_probabilities_tensor sum: ' +
#           str(label_pred_probabilities_tensor.sum(dim=1)))

#     # From the label_predicted probabilities_tensor, compute the label_name
#     label_gt_probabilities = label_gt_tensor.tolist()  # get the value from the tensor
#     label_gt_name = label_gt_probabilities.index(max(label_gt_probabilities))

#     # From the label_predicted probabilities_tensor, compute the label_name
#     label_probabilities = label_pred_probabilities_tensor.tolist()  # get the value from the tensor
#     label_pred_name = label_probabilities.index(max(label_probabilities))

#     print('Ground Truth digit: ' + str(label_gt_name))
#     print('Predicted digit: ' + str(label_pred_name))

    # ------------------------------------
    # Ex1 c)
    # ------------------------------------
    # call getitem for an idx and print the resutl
    # image_tensor, label_tensor = dataset.__getitem__(87)  # type: ignore

    # print('Image tensor shape: ' + str(image_tensor.shape))
    # print('Label tensor: ' + str(label_tensor))

    # # Display the image
    # to_pil = transforms.ToPILImage()
    # image = to_pil(image_tensor)  # get the image from the tensor

    # plt.figure()
    # plt.imshow(image, cmap='gray')

    # # Get teh value of the digit to put on the title
    # label = label_tensor.tolist()  # get the value from the tensor
    # label_name = label.index(max(label))
    # plt.title('Label ' + str(label_name))

    # plt.axis("off")
    # plt.show()


if __name__ == '__main__':
    main()
