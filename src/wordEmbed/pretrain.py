import torch
import numpy as np
import torch.nn as nn

class PreTrainEmbedding(nn.Module):
  def __init__(self, voc, pretrain_embedding_path, freeze=False, bow=True):
    super(PreTrainEmbedding, self).__init__()
    embedding_dict = dict() # idx to vector
    next_idx = len(voc.items()) + 1
    for item in voc.items():
      embedding_dict[item[1]] = np.array([])
    embedding_dict[0] = np.array([])
    self.len_of_emb = 0
    with open(pretrain_embedding_path, 'r') as f:
      lines = f.read()
      lines = lines.split('\n')
      for line in lines:
        word = line.split('\t')[0]
        if word != '#UNK#':
          word = word.lower()
        if word not in voc:
          voc[word] = next_idx
          next_idx += 1
        word_embedding = line.split('\t')[1].split(' ')
        word_embedding = list(map(lambda x: float(x), word_embedding))
        if self.len_of_emb == 0:
          self.len_of_emb = len(word_embedding)
        word_embedding = np.array(word_embedding)
        embedding_dict[voc[word]] = word_embedding
    embedding_dict[0] = np.array([0 for _ in range(self.len_of_emb)])
    unknown_idx = voc['#UNK#']
    unk_embedding = embedding_dict[unknown_idx]
    for key in embedding_dict.keys():
      if len(embedding_dict[key]) == 0:
        embedding_dict[key] = unk_embedding
    embedding_dict = list(sorted(embedding_dict.items(), key=lambda x: x[0]))
    emb_arr = [item[1] for item in embedding_dict]
    self.weight = torch.FloatTensor(emb_arr)
    self.bow = bow
    if bow:
      self.embedding = torch.nn.EmbeddingBag.from_pretrained(self.weight, freeze=freeze, mode='mean')
    else:
      self.embedding = torch.nn.Embedding.from_pretrained(self.weight, freeze=freeze)
    self.word2idx = voc
  
  def forward(self, x):
    return self.embedding(x)
