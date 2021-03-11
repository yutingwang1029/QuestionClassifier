import torch
import numpy as np

class Bilstm(torch.nn.Module):
  def __init__(self,  vocab_size, input_dim, hidden_dim):
    super(Bilstm, self).__init__()
    self.hidden_dim = hidden_dim
    self.bilstm = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True)
  def forward(self, x):
    # assert(x.shape[0] == 1)
    sent = x[0]
    length = len(sent)
    bilitm_out, _ = self.bilstm(x.view(length, x.shape[0], -1))
    temp = torch.cat((bilitm_out[0, 0, self.hidden_dim:], bilitm_out[length - 1, 0, :self.hidden_dim])).view(x.shape[0], -1)
    return temp
