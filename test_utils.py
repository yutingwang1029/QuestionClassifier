from utils import preprocessing, get_stopword, remove_stop, create_vocab
import configparser
import torch
import numpy as np
import wordEmbed

global_config_path = './config.ini'

def test_utils():
  config = configparser.ConfigParser()
  config.read(global_config_path)
  stopword_path = config["GENERAL"]["stop_word_path"]
  datapath = config["GENERAL"]["test_path"]
  sents, _ = preprocessing(datapath)
  vo, sents = create_vocab(sents, stopword_path)
  print(sents)
  vecs = []
  for sent in sents:
      vec = []
      for token in sent:
        vec.append(vo[token])
      vecs.append(vec)
  randomVec = wordEmbed.RandomWordVec()
  for i in range(len(vecs)):
    input = torch.LongTensor(vecs[i])
    print(input)
    print(randomVec.forward(input))

test_utils()