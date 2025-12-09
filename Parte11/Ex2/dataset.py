import glob
import os
import zipfile
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):

        # Store the arguments in class properties
        self.args = args
        self.train = is_train

        # ---------------------------------
        # Create the inputs
        # --------------------------------
        # create a list of image filenames to be loaded later

        # Create the image path varialbe
        print(args['dataset_folder'])
        split_name = 'train' if is_train else 'test'
        image_path = os.path.join(args['dataset_folder'], split_name, 'images/')

        print('image path is: ' + image_path)

        self.image_filenames = glob.glob(image_path + "/*.jpg")
        self.image_filenames.sort()  # Sort the filenames to ensure consistent order

        # print("image_filenames= " + str(self.image_filenames))

        # ---------------------------------
        # Create the labels
        # --------------------------------
        self.labels_filename = os.path.join(
            args['dataset_folder'], split_name, 'labels.txt')

        self.labels = []  # create a list of labels in the same order as the images

        with open(self.labels_filename, "r") as f:  # type: ignore
            for line in f:
                # print("line= " + line)
                parts = line.strip().split()   # split by whitespace
                # print('parts= ' + str(parts))
                label = float(parts[1])    # take the second column
                # print('label= ' + label)
                self.labels.append(label)

        # Select the percentage of examples specified in args
        num_examples = round(len(self.image_filenames) * args['percentage_examples'])

        # Reduce the size of the image_fileanames and labels
        self.image_filenames = self.image_filenames[0:num_examples]
        self.labels = self.labels[0:num_examples]

        # To conver from a list ot tensor. USed in the method bellow
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        # This function returns the number of examples in the dataset
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # This fucntion will get as input the idx of a example and should return the input and corresponding output for that example.
        # the return of the function should be a tuple (input, output), but the values in the tuple must be tensors
        # In other words: return (image_tensor, label_tensor)

        # ----------------------------
        # Get the label as a tensor
        # ----------------------------
        label_index = int(self.labels[idx])
        label = [0]*10  # create a list of ten zeros
        label[label_index] = 1  # set the position of the label to 1

        label_tensor = torch.tensor(label, dtype=torch.float)

        # ----------------------------
        # Get the image as a tensor
        # ----------------------------
        image_filename = self.image_filenames[idx]

        image = Image.open(image_filename).convert('L')  # make sure its loaded as a grayscale
        image_tensor = self.to_tensor(image)

        return image_tensor, label_tensor
