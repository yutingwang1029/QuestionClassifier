from torch import nn
import torch
import torch.nn.functional as F


class BiLSTM_random(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,tagset_size):
        super(BiLSTM_random,self).__init__()
        self.bilstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sen_batch, sentence):
        bilstm_out, _ = self.bilstm(sen_batch.view(len(sentence), 1, -1))
        h = torch.cat((_[0][0], _[0][1]), 1)
        tag_space = self.hidden2tag(h)
        tag_scores = F.log_softmax(tag_space.view(1, -1), dim=1)
        return tag_scores
