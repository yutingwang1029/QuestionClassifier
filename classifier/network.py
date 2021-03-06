import torch
import torch.nn as nn

import classifier

class NeuralNetworkClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNetworkClassifier, self).__init__()
    net = [
      torch.nn.Linear(input_size, hidden_size),
      torch.nn.Tanh(),
      torch.nn.Linear(hidden_size, output_size),
      torch.nn.LogSoftmax(dim = 1)
    ]
    self.model = nn.Sequential(*net)

  def forward(self, x):
    return self.model(x)