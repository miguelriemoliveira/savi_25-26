import glob
import os
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import requests
import seaborn
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm


class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        # Storing arguments in class properties
        self.args = args
        self.model = model

        # Create the dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
        # For testing we typically set shuffle to false

        # Define loss for the epochs
        self.train_epoch_losses = []
        self.test_epoch_losses = []

        # Setup loss function
        self.loss = nn.MSELoss()  # Mean Squared Error Loss

        # Define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        # Setup the figure
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])  # type: ignore
        axis.set_ylim([0, 0.1])  # type: ignore

    def train(self):

        print('Training started. Max epochs = ' + str(self.args['num_epochs']))

        # -----------------------------------------
        # Iterate all epochs
        # -----------------------------------------
        for epoch_idx in range(self.args['num_epochs']):  # number of epochs
            print('\nEpoch index = ' + str(epoch_idx))

            # -----------------------------------------
            # Train - Iterate over batches
            # -----------------------------------------
            self.model.train()  # set model to training mode
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.train_dataloader), total=num_batches):  # type: ignore

                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # Update model
                self.optimizer.zero_grad()  # resets the gradients from previous batches
                batch_loss.backward()  # the actual backpropagation
                self.optimizer.step()

            # -----------------------------------------
            # Test - Iterate over batches
            # -----------------------------------------
            self.model.eval()  # set model to evaluation mode

            test_batch_losses = []
            num_batches = len(self.test_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.test_dataloader), total=num_batches):  # type: ignore
                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # During test there is no model update

            # ---------------------------------
            # End of the epoch training
            # ---------------------------------
            print('Finished epoch ' + str(epoch_idx) + ' out of ' + str(self.args['num_epochs']))
            # print('batch_losses: ' + str(batch_losses))

            # update the training epoch losses
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            # update the testing epoch losses
            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            # Draw the updated training figure
            self.draw()

        print('Training completed.')
        print('Training losses: ' + str(self.train_epoch_losses))
        print('Test losses: ' + str(self.test_epoch_losses))

    def draw(self):

        # plot training
        xs = range(1, len(self.train_epoch_losses)+1)
        ys = self.train_epoch_losses
        plt.plot(xs, ys, 'r-', linewidth=2)

        # plot testing
        xs = range(1, len(self.test_epoch_losses)+1)
        ys = self.test_epoch_losses
        plt.plot(xs, ys, 'b-', linewidth=2)

        plt.legend(['Train', 'Test'])

        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))

    def evaluate(self):

        # -----------------------------------------
        # Iterate over test batches and compute the ground trutch and predicted  values for all examples
        # -----------------------------------------
        self.model.eval()  # set model to evaluation mode
        num_batches = len(self.test_dataloader)

        self.gts = []  # list of ground truth labels
        self.preds = []  # list of predicted labels

        gt_classes = []
        predicted_classes = []
        for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                enumerate(self.test_dataloader), total=num_batches):  # type: ignore

            batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()

            # Compute the predicted labels
            label_pred_tensor = self.model.forward(image_tensor)

            # Compute the probabilities using softmax
            label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)
            batch_predicted_classes = label_pred_probabilities_tensor.argmax(dim=1).tolist()

            # print('batch_gt_classes: ' + str(batch_gt_classes))
            # print('batch_predicted_classes: ' + str(batch_predicted_classes))

            gt_classes.extend(batch_gt_classes)
            predicted_classes.extend(batch_predicted_classes)

        print('Ground truth classes: ' + str(gt_classes))
        print('Predicted classes: ' + str(predicted_classes))

        # -----------------------------------------
        # Create the confusion matrix
        # -----------------------------------------
        confusion_matrix = np.zeros((10, 10), dtype=int)

        for gt_class, predicted_class in zip(gt_classes, predicted_classes):
            confusion_matrix[gt_class][predicted_class] += 1

        # -----------------------------------------
        # Draw the confusion matrix
        # -----------------------------------------
        plt.figure(2)
        class_names = [str(i) for i in range(10)]
        title = 'Confusion Matrix'
        seaborn.heatmap(confusion_matrix,
                        annot=True,       # Anotar as células com os valores
                        fmt='d',          # Formato dos números (inteiros para contagens)
                        # Mapa de cores (pode escolher outro, ex: 'viridis', 'YlGnBu')
                        cmap='Blues',
                        cbar=True,        # Mostrar barra de cores
                        xticklabels=class_names,  # Rótulos do eixo X (classes previstas)
                        yticklabels=class_names)  # Rótulos do eixo Y (classes verdadeiras)

        plt.title(title, fontsize=16)  # Título do gráfico
        plt.xlabel('Predicted classes', fontsize=14)  # Rótulo do eixo X
        plt.ylabel('True classes', fontsize=14)  # Rótulo do eixo Y
        plt.xticks(rotation=0, ha='right', fontsize=12)  # Rodar rótulos do X para melhor leitura
        plt.yticks(rotation=0, fontsize=12)  # Rótulos do Y
        plt.tight_layout()  # Ajusta o layout para evitar sobreposições

        plt.savefig(os.path.join(self.args['experiment_full_name'],
                                 'confusion_matrix.png'))

        # -----------------------------------------
        # Compute TPs, FPs, TNs, FNs for each class
        # -----------------------------------------
        statistics = {}

        for i in range(10):

            TPs = int(confusion_matrix[i][i])
            FPs = int(sum(confusion_matrix[:, i]) - TPs)
            FNs = int(sum(confusion_matrix[i, :]) - TPs)
            precision, recall = self.getPrecisionRecall(TPs, FPs, FNs)

            d = {'digit': i, 'TPs': TPs, 'FPs': FPs, 'FNs': FNs,
                 'precision': precision, 'recall': recall}
            statistics[i] = d

        print('Statistics per class: ' + str(statistics))

        # -----------------------------------------
        # Write the dictionary to a json file
        # -----------------------------------------
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(statistics, f, indent=4)

    def getPrecisionRecall(self, TPs, FPs, FNs):

        den = TPs + FPs
        if den == 0:
            precision = None
        else:
            precision = TPs / (TPs + FPs)

        den = TPs + FNs
        if den == 0:
            recall = None
        else:
            recall = TPs / (TPs + FNs)

        return precision, recall
