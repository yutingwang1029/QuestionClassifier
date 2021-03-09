
import torch.nn as nn
import torch

torch.manual_seed(0)

class RandomWordVec(nn.Module):
    def __init__(self, voc_size=8352, dim=3, bow=True):
        super(RandomWordVec, self).__init__()
        if bow == False:
            self.bow = False
            self.embedding = nn.Embedding(voc_size+1, dim)
        else:
            self.bow = True
            self.embedding = nn.EmbeddingBag(voc_size+1, dim, mode='mean')

    def forward(self, x):
        # if x.shape[0] == 1:
        #     offset = torch.LongTensor([0])
        #     temp = self.embedding(x[0], offset)
        #     return torch.Tensor(temp)
        # print(x)
        return self.embedding(x)
  

