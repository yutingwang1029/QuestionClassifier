from model import QuestionClassifier
import wordEmbed
import utils
import configparser
import torch
import numpy as np

global_config_path = './config.ini'

def test_model():
  config = configparser.ConfigParser()
  config.read(global_config_path)
  stopword_path = config["GENERAL"]["stop_word_path"]
  datapath = config["GENERAL"]["test_path"]
  # pretrain_path = config["WORD_EMBED"]["pretrain_path"]
  x, y = utils.preprocessing(datapath)
  voc, sents = utils.create_vocab(x, stopword_path)
  # vecs = utils.word2vec(voc, sents)
  label2idx, _ = utils.get_label_dict(y)
  tags = utils.get_tags(y, label2idx)
  clf = QuestionClassifier(
    bow=False,
    voc=voc,
    pretrain_embedding_path='',
    freeze=False,
    random_or_word2vec='random',
    bilstm_input_dim=3,
    bilstm_hidden_dim=20,
    bilstm_max_len=11,
    nn_input_dim=3,
    nn_hidden_dim=100,
    nn_output_size=len(label2idx)
  )
  output = clf.forward(sents)
  print(output)


test_model()