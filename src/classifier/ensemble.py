import torch
import torch.nn as nn

class EnsembleClassifier(nn.Module):
  def __init__(self, input_size, hidden_dim_1, output_size, ensemble_size=3):
    super(EnsembleClassifier, self).__init__()
    net = [
      torch.nn.Linear(input_size, hidden_dim_1),
      torch.nn.Tanh(),
      torch.nn.Linear(hidden_dim_1, output_size),
    ]
    self.ens_size = ensemble_size
    if self.ens_size == 1:
      self.model = nn.Sequential(*net)
    else:
      self.model = []
      for _i in range(self.ens_size):
        self.model.append(nn.Sequential(*net))

  def forward(self, x):
    if self.ens_size == 1:
      return self.model(x)
    else:
      ret = self.model[0](x)
      print(ret)
      for i in range(1, self.ens_size):
        ret += self.model[i](x)
      print(ret)
      print('---')
      return ret / self.ens_size