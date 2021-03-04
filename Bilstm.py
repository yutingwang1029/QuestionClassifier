from torch import nn
import torch
import torch.nn.functional as F

#bilstm model using random vector
class BiLSTM_random(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size):
        super(BiLSTM_random,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        bilstm_out, _ = self.bilstm(embeds.view(len(sentence), 1, -1))
        h = torch.cat((_[0][0], _[0][1]), 1)
        tag_space = self.hidden2tag(h)
        tag_scores = F.log_softmax(tag_space.view(1, -1), dim=1)
        return tag_scores

#bilstm model using pretrain weight
class BiLSTM_pre(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, weight, tagset_size,freeze=False):
        super(BiLSTM_pre, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding.from_pretrained(weight, freeze=freeze) #vocab_size=weights
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        bilstm_out, _ = self.bilstm(embeds.view(len(sentence), 1, -1))
        h = torch.cat((_[0][0], _[0][1]), 1)
        tag_space = self.hidden2tag(h)
        tag_scores = F.log_softmax(tag_space.view(1, -1), dim=1)
        return tag_scores

