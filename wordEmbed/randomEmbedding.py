
# def create_word_enbedding(encoded_sentencesword):
#     word_vectors = [i for i in range(len(encoded_sentences))]
#     emb_dim = 
#     for i in range(len(encoded_sentences)):
#         emb_layer = nn.Embedding(len(encoded_sentences[i]), emb_dim)
#         word_vectors[i] = emb_layer(torch.LongTensor(encoded_sentences[i]))
#     return word_vectors

import torch.nn as nn

class RandomWordVec(nn.Module):
    def __init__(self):
        super(RandomWordVec, self).__init__()
        self.embedding = nn.Embedding(8352, 3)

    def forward(self, x):
        return self.embedding(x)
  

