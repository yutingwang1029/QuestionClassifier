from sentVect import BiLSTM
import wordEmbed
import utils
import configparser
import torch
import numpy as np

global_config_path = './config.ini'

def test_bilstm():
  config = configparser.ConfigParser()
  config.read(global_config_path)
  stopword_path = config["GENERAL"]["stop_word_path"]
  datapath = config["GENERAL"]["test_path"]
  # pretrain_path = config["WORD_EMBED"]["pretrain_path"]
  x, _ = utils.preprocessing(datapath)
  voc, sents = utils.create_vocab(x, stopword_path)
  print(sents)
  vecs = utils.word2vec(voc, sents)
  print(vecs)
  max_len = max(list(map(lambda vec: len(vec), vecs)))
  padded_vecs, origin_lens = utils.padding(vecs, max_len)
  print(padded_vecs, origin_lens)
  emb = wordEmbed.RandomWordVec(voc_size=len(voc), dim=3, bow=False)
  sents_tensor = torch.LongTensor(padded_vecs)
  len_tensor = torch.LongTensor(origin_lens)
  output = emb.forward(sents_tensor)
  bilstm = BiLSTM(3, 10, 3, max_len)
  bilstm_output = bilstm.forward(output, len_tensor)
  print(bilstm_output)
  

test_bilstm()