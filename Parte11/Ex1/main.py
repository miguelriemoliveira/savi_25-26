#!/usr/bin/env python3
# shebang line for linux / mac

import glob
import os
from random import randint
import shutil
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse

import torch
from dataset import Dataset

from torchvision import transforms
from model import Model
from trainer import Trainer
from datetime import datetime


def main():

    # ------------------------------------
    # Setu pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('-df', '--dataset_folder', type=str,
                        default='/home/mike/data/savi_datasets/mnist')
    parser.add_argument('-pe', '--percentage_examples', type=float, default=0.2,
                        help='Percentage of examples to use for training and testing')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size for training and testing.')
    parser.add_argument('-ep', '--experiment_path', type=str,
                        default='/home/mike/data/savi_experiments',
                        help='Path to save experiment results.')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Create the experiment
    # ------------------------------------

    # experiment_name = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    experiment_name = datetime.today().strftime('%Y-%m-%d %H')  # same experiment every hour

    args['experiment_full_name'] = os.path.join(
        args['experiment_path'], experiment_name)

    print('Starting experiment: ' + args['experiment_full_name'])

    if os.path.exists(args['experiment_full_name']):
        shutil.rmtree(args['experiment_full_name'])
        print('Experiment folder already exists. Deleting to start fresh.')

    os.makedirs(args['experiment_full_name'])

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

    trainer.evaluate()  # run evaluation


if __name__ == '__main__':
    main()
