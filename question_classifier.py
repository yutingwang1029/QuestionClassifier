import configparser
import torch.optim as optim
from model import QuestionClassifier
import torch.nn as nn
#from sklearn.metrics import mean_squared_error
from config import get_config
import utils
import torch
from dataloader import DataLoader
import numpy as np

global_config_path = './config.ini'

def cmdparser():
  # todo: parse the cmd argument
  pass

def train_val(model, test_x, test_y):
  y_preds = list()
  y_real = list()
  with torch.no_grad():
    for j in range(len(test_x)):
      predict = model([test_x[j]])
      y_preds.extend(predict.argmax(dim=1).numpy().tolist())
      y_real.extend([test_y[j]])
  return np.sum(np.array(y_preds)==y_real)/len(y_real)

def train(config, voc, label_num, dataloader):
  model = QuestionClassifier(
    bow=config['bow'] == 'True',
    voc=voc,
    pretrain_embedding_path=config['pretrain_embedding_path'],
    freeze=config['freeze'] == 'True',
    random_or_word2vec=config['random_or_word2vec'],
    bilstm_input_dim=config['bilstm_input_dim'],
    bilstm_hidden_dim=config['bilstm_hidden_dim'],
    bilstm_max_len=config['bilstm_max_len'],
    nn_input_dim=config['nn_input_dim'],
    nn_hidden_dim_1=config['nn_hidden_dim_1'],
    nn_hidden_dim_2=config['nn_hidden_dim_2'],
    nn_output_size=label_num
  )
  print(model)
  test_x, test_y = dataloader.get_test_data()
  test_y = test_y
  optimizer = optim.SGD(model.parameters(), lr=float(config['lr']), weight_decay=float(config['lr']))
  criterion = nn.CrossEntropyLoss()
  model.train()
  for i in range(int(config['epoches'])):
    batches = dataloader.length // int(config['batch_size'])
    for _j in range(batches):
      features, labels = dataloader.next_batch()
      labels = torch.LongTensor(labels)
      optimizer.zero_grad()
      probs = model(features)
      loss = criterion(probs, labels)
      loss.backward()
      optimizer.step()
    # if i % 5 == 0 or i == int(config['epoches']) - 1:
    print(f"----- epoch {i} -----")
    acc = train_val(model, test_x, test_y)
    print(f"epoch {i} finished, acc: {acc}")
  return model

def test():
  pass

if __name__ == "__main__":
  config_dict = get_config(global_config_path)
  x, y = utils.preprocessing(config_dict['data_path'])
  voc, sents = utils.create_vocab(x, config_dict['stop_word_path'])
  label2idx, _ = utils.get_label_dict(y)
  dataloader = DataLoader(sents, y, int(config_dict['batch_size']), shuffle=True, test_ratio=0.2, label2idx=label2idx)
  train(config_dict, voc, len(label2idx), dataloader)