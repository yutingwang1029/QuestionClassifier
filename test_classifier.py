from classifier import NeuralNetworkClassifier

import torch

def test_classifier():
  test_input = torch.Tensor([[1,2,3,4,5,6], [7,8,9,10,11,12]])
  nn = NeuralNetworkClassifier(6, 20, 3)
  temp = nn.forward(test_input)
  print(temp)


test_classifier()
