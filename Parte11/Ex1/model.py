
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()  # call the parent constructor

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define the layers of the model
        self.fc = nn.Linear(ninputs, noutputs)

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')

    def forward(self, x):

        # print('Forward method called ...')
        # print('Input x.shape = ' + str(x.shape))

        # flatten the input to a vector of 1x28x28
        x = x.view(x.size(0), -1)
        # print('Input x.shape = ' + str(x.shape))

        # Now we can pass through the fully connected layer
        y = self.fc(x)
        # print('Output y.shape = ' + str(y.shape))

        return y

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
