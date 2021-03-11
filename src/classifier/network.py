import torch

class NeuralNetwork(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(NeuralNetwork, self).__init__()
    net = [
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, output_dim)
    ]
    self.model = torch.nn.Sequential(*net)
  
  def forward(self, x):
    return self.model(x)