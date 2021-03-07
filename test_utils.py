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
  print(vo)
  vecs = []
  for sent in sents:
      vec = []
      for token in sent:
          if token in vo:
            vec.append(vo[token])
      vecs.append(vec)
  randomVec = wordEmbed.RandomWordVec(bow=False)
  for i in range(len(vecs)):
    input = torch.LongTensor(vecs[i])
    #print(input)
    temp = randomVec.forward(input)
    sum_of_tensor = temp[0] 
    for i in range(1, len(temp)):
      sum_of_tensor += temp[i]
    sum_of_tensor /= len(temp)
    #print(sum_of_tensor)
  # for i in range(len(vecs)):
  #   input = torch.LongTensor(vecs[i])
  #   print(input)
  #   temp = randomVec.forward(input)
  #   print(temp)

test_utils()