import torch

from classifier import NeuralNetwork
from sentVect import BOW, Bilstm, BowBilstm

class QuestionClassifier(torch.nn.Module):
  def __init__(
    self,
    bow,
    bilstm,
    vocab_size,
    from_pretrain=False,
    pre_train_weight=None,
    freeze=False,
    embedding_dim=300,
    bilstm_hidden_dim=150,
    input_dim=300,
    hidden_dim=128,
    output_dim=50
  ):
    super(QuestionClassifier, self).__init__()
    if bow == True and bilstm == False:
      self.sent_vec = BOW(
        vocab_size,
        embedding_dim,
        from_pretrain,
        pre_train_weight,
        freeze
      )
    elif bow == False and bilstm == True:
      if from_pretrain:
        emb = torch.nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze)
      else:
        emb = torch.nn.Embedding(vocab_size, embedding_dim)
      self.sent_vec = torch.nn.Sequential(
        emb,
        Bilstm(vocab_size, embedding_dim, bilstm_hidden_dim)
      )
    elif bow == True and bilstm == True:
      self.sent_vec = BowBilstm(
        vocab_size,
        embedding_dim,
        from_pretrain,
        pre_train_weight,
        freeze,
        bilstm_hidden_dim
      )
    self.classifier = NeuralNetwork(input_dim, hidden_dim, output_dim)
  
  def forward(self, x):
    return self.classifier(self.sent_vec(x))
