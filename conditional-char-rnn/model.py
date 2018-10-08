import torch
import torch.nn as nn
from torch.autograd import Variable

import config

# Creating the Network

class RNN(nn.Module):
    def __init__(self, category_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.category_size = category_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(category_size + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(category_size + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax()

        self.to(config.HOST_DEVICE)
    
    def forward(self, category, input, hidden):
        category = category.to(config.HOST_DEVICE)
        input = input.to(config.HOST_DEVICE)
        hidden = hidden.to(config.HOST_DEVICE)
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size)).to(config.HOST_DEVICE)


