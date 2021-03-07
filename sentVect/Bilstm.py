from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, max_len):
        super(BiLSTM,self).__init__()
        self.input_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.sen_len = max_len
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])))
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])))
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))) * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long())

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, sen_batch, sen_lengths):
        """
        description:
            forwarding input from embedding layer
        params:
            sen_batch: batch of sentence vectors
            sen_lengths: length for each sentences
        """
        batch_size = len(sen_batch)
        bilstm_out, _ = self.bilstm(sen_batch.view(batch_size, -1, self.input_dim))
        sen_rnn = bilstm_out.contiguous().view(batch_size, -1, 2 * self.hidden_dim)
        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)
        representation = sentence_batch
        out = self.linear(representation)
        # out_prob = F.softmax(out.view(batch_size, -1), dim=1)

        return out
