
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, Ni, No, hidden_layer_params=None, act_fn=None, dropout=0):
        '''

        :param Ni: Number of input units
        :param No: Number of output units
        :param hidden_layer_params: dictionary containing the number of hidden units per hidden layer
        :param act_fn: Activation function (default is ReLU).
        '''
        super().__init__()  # initialise parent class

        if hidden_layer_params is None:
            n_hidden_layers = 0
            Nout = No
        else:
            n_hidden_layers = len(hidden_layer_params) - 1
            Nout = hidden_layer_params[0]

        if act_fn is None:
            act_fn = nn.ReLU

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=Ni, out_features=Nout))
        self.layers.append(act_fn())

        # add hidden layers
        hidden_layers = 0
        while hidden_layers < n_hidden_layers:
            Nin = hidden_layer_params[hidden_layers]
            Nout = hidden_layer_params[hidden_layers + 1]
            self.layers.append(nn.Linear(in_features=Nin, out_features=Nout))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(act_fn())
            hidden_layers += 1

        # add output layer
        self.layers.append(nn.Linear(in_features=Nout, out_features=No))

    def forward(self, x):
        '''
        forward pass in the network
        :param x: input data
        :return: predicted values given the input data
        '''
        x = x.view(x.size(0), -1)

        for layer in self.layers:
            x = layer(x)
        return x



