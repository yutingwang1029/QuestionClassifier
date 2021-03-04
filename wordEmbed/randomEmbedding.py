
# def create_word_enbedding(encoded_sentencesword):
#     word_vectors = [i for i in range(len(encoded_sentences))]
#     emb_dim = 
#     for i in range(len(encoded_sentences)):
#         emb_layer = nn.Embedding(len(encoded_sentences[i]), emb_dim)
#         word_vectors[i] = emb_layer(torch.LongTensor(encoded_sentences[i]))
#     return word_vectors

import torch.nn as nn
import torch

torch.manual_seed(0)

class RandomWordVec(nn.Module):
    def __init__(self, voc_size=8352, dim=3, bow=True):
        super(RandomWordVec, self).__init__()
        if bow == False:
            self.bow = False
            self.embedding = nn.Embedding(voc_size, dim)
        else:
            self.bow = True
            self.embedding = nn.EmbeddingBag(voc_size, dim, mode='mean')

    def forward(self, x):
        if self.bow:
            offset = torch.LongTensor([0])
            return self.embedding(x, offset)
        else:
            return self.embedding(x)
  

