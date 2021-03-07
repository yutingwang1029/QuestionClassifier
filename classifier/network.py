import torch
import torch.nn as nn

import classifier

class NeuralNetworkClassifier(nn.Module):
  def __init__(self, input_size, hidden_dim_1, hidden_dim_2, output_size):
    super(NeuralNetworkClassifier, self).__init__()
    # net = [
    #   torch.nn.Linear(input_size, hidden_dim_1),
    #   torch.nn.Tanh(),
    #   torch.nn.Linear(hidden_dim_1, hidden_dim_2),
    #   torch.nn.ReLU(),
    #   torch.nn.Linear(hidden_dim_2, hidden_dim_1),
    #   torch.nn.Tanh(),
    #   torch.nn.Linear(hidden_dim_1, output_size)
    #   # torch.nn.Softmax(dim = 1)
    # ]
    net = [
      torch.nn.Linear(input_size, hidden_dim_1),
      torch.nn.Tanh(),
      torch.nn.Linear(hidden_dim_1, output_size)
    ]
    self.model = nn.Sequential(*net)

  def forward(self, x):
    return self.model(x)