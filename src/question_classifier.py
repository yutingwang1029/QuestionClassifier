import torch
import numpy as np
import random
import re
import argparse
import utils
from model import QuestionClassifier
from dataloader import DataLoader
from config import get_config

global_config_path = './config.ini'

def cmdparser():
  parser = argparse.ArgumentParser(description='Question Classifier')
  parser.add_argument('--preprocess', action='store_true', help='preprocess the data before training')
  parser.add_argument('--dev', action="store_true", help='dev training mode without saving the model')
  parser.add_argument('--train', action="store_true", help='train and save the model')
  parser.add_argument('--test', action="store_true", help='test existing model')
  parser.add_argument('--search', action='store_true', help="searching for hyper-params")
  parser.add_argument('--config', type=str, default=global_config_path, help="configuration path")
  args = parser.parse_args()
  return args

def test_trec(model, dataloader):
  acc = 0
  reals, preds = [], []
  for _i in range(dataloader.length):
    feat, label = dataloader.get_batch()
    output = model(feat)
    _, pred = torch.max(output.data, 1)
    reals.append(label)
    preds.append(pred)
    if label == pred:
      acc += 1
  acc_rate = float(acc) / float(dataloader.length)
  return acc_rate, reals, preds

def train(config, vocabulary, labels, stop_words, save_path='', mode='dev'):
  if config['from_pretrain'] == 'True':
    pretrain_dict = utils.load_pre_train(config['pretrain_embedding_path'])
    pretrain_weight = utils.create_word_embedding(pretrain_dict, vocabulary)
    embedding_dim = len(pretrain_weight[0])
  else:
    pretrain_weight = [0]
    embedding_dim = int(config['embedding_dim'])
  
  model = QuestionClassifier(
    bow=config['bow'] == 'True',
    bilstm=config['bilstm'] == 'True',
    vocab_size=len(vocabulary),
    embedding_dim=embedding_dim,
    from_pretrain=config['from_pretrain'] == 'True',
    pre_train_weight=torch.FloatTensor(pretrain_weight),
    freeze=config['freeze'] == 'True',
    bilstm_hidden_dim=int(config['bilstm_hidden_dim']),
    input_dim=int(config['input_dim']),
    hidden_dim=int(config['hidden_dim']),
    output_dim=len(labels)
  )
  print(model)
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']))
  if mode == 'dev':
    train_set = utils.load_data(config['train_path'])
    val_set = utils.load_data(config['dev_path'])
    val_loader = DataLoader(vocabulary, labels, stop_words, val_set, 1, False)
  else:
    train_set = utils.load_data(config['raw_path'])
  test_set = utils.load_data(config['test_path'])
  train_loader = DataLoader(vocabulary, labels, stop_words, train_set, int(config['batch_size']), config['padding'] == 'True', int(config['padding_len']))
  test_loader = DataLoader(vocabulary, labels, stop_words, test_set, 1, config['padding'] == 'True')

  for i in range(int(config['epochs'])):
    error = 0
    steps= train_loader.length // int(config['batch_size'])
    for _ in range(steps):
      feat, target = train_loader.get_batch()
      optimizer.zero_grad()
      preds = model(feat)
      loss = loss_function(preds, target)
      error += loss.item()
      loss.backward()
      optimizer.step()
    if mode == 'train':
      trec_acc, _, _ = test_trec(model, test_loader)
      print(f'--- epoch {i+1} ---')
      print(f'loss: {error / len(train_set)}, TREC acc: {trec_acc}')
    elif mode == 'dev':
      val_acc, _, _ = test_trec(model, val_loader)
      trec_acc, _, _ = test_trec(model, test_loader)
      print(f'--- epoch {i+1} ---')
      print(f'loss: {error / len(train_set)}, validation acc: {val_acc}, TREC acc: {trec_acc}')
  if mode == 'train' and save_path != '':
    torch.save(model.state_dict(), save_path)
  return model

def test(config, vocabulary, labels, stop_words, save_path):
  if config['from_pretrain'] == 'True':
    pretrain_dict = utils.load_pre_train(config['pretrain_embedding_path'])
    pretrain_weight = utils.create_word_embedding(pretrain_dict, vocabulary)
    embedding_dim = len(pretrain_weight[0])
  else:
    pretrain_weight = [0]
    embedding_dim = int(config['embedding_dim'])
  
  model = QuestionClassifier(
    bow=config['bow'] == 'True',
    bilstm=config['bilstm'] == 'True',
    vocab_size=len(vocabulary),
    embedding_dim=embedding_dim,
    from_pretrain=config['from_pretrain'] == 'True',
    pre_train_weight=torch.FloatTensor(pretrain_weight),
    freeze=config['freeze'] == 'True',
    bilstm_hidden_dim=int(config['bilstm_hidden_dim']),
    input_dim=int(config['input_dim']),
    hidden_dim=int(config['hidden_dim']),
    output_dim=len(labels)
  )
  test_set = utils.load_data(config['test_path'])
  test_loader = DataLoader(vocabulary, labels, stop_words, test_set, 1, config['padding'] == 'True')
  if int(config['ensemble_size']) == 1:
    model.load_state_dict(torch.load(save_path[0]))
    model.eval()
    trec_acc, reals, final_preds = test_trec(model, test_loader)
  else:
    final_preds = []
    weight = []
    y_preds = []
    for path in save_path:
      model.load_state_dict(torch.load(path))
      model.eval()
      trec_acc, reals, preds = test_trec(model, test_loader)
      for idx in range(len(preds)):
        if len(y_preds) <= idx:
          y_preds.append([preds[idx].numpy()[0]])
          weight.append([trec_acc])
        else:
          y_preds[idx].append(preds[idx].numpy()[0])
          weight[idx].append(trec_acc)
    y_real = np.array(reals)
    for idx in range(len(y_preds)):
      # each class
      votes = dict([[i, 0] for i in range(len(labels))])
      for j in range(len(y_preds[idx])):
        votes[y_preds[idx][j]] += weight[idx][j]
      final_preds.append(list(sorted(votes.items(), key=lambda x: x[1], reverse=True))[0][0])
    trec_acc = np.sum(np.array(final_preds) == y_real) / len(y_real)

  results = []
  for idx in range(len(final_preds)):
    results.append((labels[reals[idx]], labels[final_preds[idx]]))
  string = ''
  for line in results:
    string += f'real label: {line[0]}, predict label: {line[1]}\n'
  string += f'TREC 10 accuracy: {trec_acc*100}%'
  with open(config['output_path'], 'w') as f:
    f.write(string)


def main():
  args = cmdparser()
  config = get_config(args.config)
  if args.preprocess:
    utils.preprocess(
      config['raw_path'],
      config['train_path'],
      config['dev_path'],
      config['label_path'],
      config['stop_word_path'],
      config['vocabulary_path']
    )
  labels = utils.load_labels(config['label_path'])
  vocabulary = utils.load_vocabulary(config['vocabulary_path'])
  stop_words = utils.load_stop_words(config['stop_word_path'])
  
  if args.dev:
    train(config, vocabulary, labels, stop_words, save_path='', mode='dev')
  elif args.train:
    if int(config['ensemble_size']) == 1:
      train(config, vocabulary, labels, stop_words, save_path=config['model_path'], mode='train')
    else:
      for i in range(int(config['ensemble_size'])):
        train(config, vocabulary, labels, stop_words, save_path=config[f'model_path_{i+1}'], mode='train')
  elif args.test:
    if int(config['ensemble_size']) == 1:
      test(config, vocabulary, labels, stop_words, save_path=[config['model_path']])
    else:
      test_paths = [config[f'model_path_{i+1}'] for i in range(int(config['ensemble_size']))]
      test(config, vocabulary, labels, stop_words, save_path=test_paths)
      
if __name__ == "__main__":
  main()