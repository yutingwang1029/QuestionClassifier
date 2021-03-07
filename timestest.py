from utils import preprocessing, get_stopword, remove_stop, create_vocab
import configparser
import torch
import numpy as np
import wordEmbed
#import word2sen

global_config_path = './config.ini'

def test_utils():
  config = configparser.ConfigParser()
  config.read(global_config_path)
  stopword_path = config["GENERAL"]["stop_word_path"]
  datapath = config["GENERAL"]["test_path"]
  x, y = preprocessing(datapath)
  stopword_list = get_stopword(stopword_path)
  sents = remove_stop(x, stopword_list)
  vo = create_vocab(sents)
  print(vo)
  for sent in sents:
    for token in sent:
        print(vo[token])
  
    
    
test_utils()

