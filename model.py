from wordEmbed import RandomWordVec, PreTrainEmbedding
from sentVect import BiLSTM
from classifier import NeuralNetworkClassifier
import utils

import torch
import torch.nn as nn

class QuestionClassifier(nn.Module):
  def __init__(
    self, 
    bow=True, 
    voc={},
    pretrain_embedding_path='',
    freeze=False,
    random_or_word2vec='random',
    bilstm_input_dim=3,
    bilstm_hidden_dim=20,
    bilstm_max_len=40,
    nn_input_dim=10,
    nn_hidden_dim=100,
    nn_output_size=10
  ):
    super(QuestionClassifier, self).__init__()
    if random_or_word2vec == 'random':
      config = {
        "voc_size": len(voc),
        "dim": bilstm_input_dim,
        "bow": bow
      }
      self.embedding = RandomWordVec(**config)
    else:
      config = {
        "voc": voc,
        "pretrain_embedding_path": pretrain_embedding_path,
        "freeze": freeze,
        "bow": bow
      }
      self.embedding = PreTrainEmbedding(**config)
    self.bow = bow
    self.max_len = bilstm_max_len
    self.voc = voc
    if bow == False:
      self.bilstm = BiLSTM(bilstm_input_dim, bilstm_hidden_dim, nn_input_dim, bilstm_max_len)
    self.classifier = NeuralNetworkClassifier(nn_input_dim, nn_hidden_dim, nn_output_size)
  
  def forward(self, x):
    """
    description:
      x: sentence batch
    """
    vecs = utils.word2vec(self.voc, x)
    padded_vecs, origin_lens = utils.padding(vecs, self.max_len)
    sents_tensor = torch.LongTensor(padded_vecs)
    len_tensor = torch.LongTensor(origin_lens)
    emb = self.embedding(sents_tensor)
    if not self.bow:
      sent_vecs = self.bilstm(emb, len_tensor)
      return self.classifier(sent_vecs)
    else:
      # print(emb)
      ret = self.classifier(emb)
      return ret