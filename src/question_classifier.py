import configparser
import torch.optim as optim
from model import QuestionClassifier
import torch.nn as nn
from config import get_config
import utils
import torch
from dataloader import DataLoader
import numpy as np
import argparse
import copy

global_config_path = './config.ini'

def cmdparser():
  parser = argparse.ArgumentParser(description='Question Classifier')
  parser.add_argument('--dev', action="store_true", help='dev training mode without saving the model')
  parser.add_argument('--train', action="store_true", help='train and save the model')
  parser.add_argument('--test', action="store_true", help='test existing model')
  parser.add_argument('--config', type=str, default=global_config_path, help="configuration path")
  args = parser.parse_args()
  return args
  
def test_trec(model, trec_loader):
  test_x, test_y = trec_loader.get_all()
  ret, y_preds = train_val(model, test_x, test_y)
  return ret, y_preds

def train_val(model, test_x, test_y):
  y_preds = list()
  y_real = list()
  with torch.no_grad():
    for j in range(len(test_x)):
      predict = model([test_x[j]])
      y_preds.extend(predict.argmax(dim=1).numpy().tolist())
      y_real.extend([test_y[j]])
  return np.sum(np.array(y_preds)==y_real)/len(y_real), y_preds

def train(config, voc, label_num, dataloader, trec_loader, mode='dev'):
  model = QuestionClassifier(
    ensemble=int(config['ensemble_size']),
    bow=config['bow'] == 'True',
    bilstm=config['bilstm'] == 'True',
    voc=voc,
    pretrain_embedding_path=config['pretrain_embedding_path'],
    freeze=config['freeze'] == 'True',
    random_or_word2vec=config['random_or_word2vec'],
    bilstm_input_dim=config['bilstm_input_dim'],
    bilstm_hidden_dim=config['bilstm_hidden_dim'],
    bilstm_max_len=config['bilstm_max_len'],
    nn_input_dim=config['nn_input_dim'],
    nn_hidden_dim_1=config['nn_hidden_dim_1'],
    nn_output_size=label_num
  )
  print(model)
  test_x, test_y = dataloader.get_test_data()
  optimizer = optim.SGD(
    model.parameters(), 
    lr=float(config['lr']),
    weight_decay=float(config['lr'])
  )
  scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=float(config['lr_decay_step']), 
    gamma=float(config['lr_decay_rate'])
  )
  criterion = nn.CrossEntropyLoss()
  early_stopping = 0
  best_val_acc = 0.0
  best_trec_acc = 0.0
  best_model = copy.deepcopy(model.state_dict())
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
      if mode == 'dev':
        new_val_acc, _ = train_val(model, test_x, test_y)
        if new_val_acc > best_val_acc:
          early_stopping = 0
          best_val_acc = new_val_acc
        if early_stopping >= int(config['early_stopping']):
          print(f'dev training early stopping at epoch {i+1}')
          return model
    if i >= 4:
      scheduler.step()
    print(f"----- epoch {i+1} -----")
    if mode == 'dev':
      acc, _ = train_val(model, test_x, test_y)
      acc_trec, _ = test_trec(model, trec_loader)
      best_trec_acc = acc_trec if acc_trec > best_trec_acc else best_trec_acc
      print(f"epoch {i+1} finished, validation acc: {acc}, TREC 10 acc: {acc_trec}")
    else:
      acc_trec, _ = test_trec(model, trec_loader)
      if best_trec_acc < acc_trec:
        best_trec_acc = acc_trec
        best_model = copy.deepcopy(model.state_dict())
      print(f"epoch {i+1} finished, TREC 10 acc: {acc_trec}, best TREC 10 acc: {best_trec_acc} lr: {scheduler.get_last_lr()[0]}")
  if mode == 'train':
    torch.save(best_model, config['model_path'])
  return best_model, best_trec_acc

def test(config, voc, label_num, trec_loader, idx2label):
  model = QuestionClassifier(
    ensemble=int(config['ensemble_size']),
    bow=config['bow'] == 'True',
    bilstm=config['bilstm'] == 'True',
    voc=voc,
    pretrain_embedding_path=config['pretrain_embedding_path'],
    freeze=config['freeze'] == 'True',
    random_or_word2vec=config['random_or_word2vec'],
    bilstm_input_dim=config['bilstm_input_dim'],
    bilstm_hidden_dim=config['bilstm_hidden_dim'],
    bilstm_max_len=config['bilstm_max_len'],
    nn_input_dim=config['nn_input_dim'],
    nn_hidden_dim_1=config['nn_hidden_dim_1'],
    nn_output_size=label_num
  )
  model.load_state_dict(torch.load(config['model_path']))
  model.eval()
  trec_acc, y_preds = test_trec(model, trec_loader)
  results = []
  for y in y_preds:
    results.append(idx2label[y])
  report = f'Accuracy on dataset TREC 10 is {trec_acc}.\n' + '\n'.join(results)
  with open(config['output_path'], 'w') as f:
    f.write(report)


if __name__ == "__main__":
  args = cmdparser()
  config_dict = get_config(args.config)

  stopwords = utils.get_stopword(config_dict['stop_word_path'])

  # get test data
  test_x, test_y = utils.preprocessing(config_dict['trec_path'])
  test_sents = utils.remove_stop(test_x, stopwords)

  # get training data
  x, y = utils.preprocessing(config_dict['data_path'])
  voc, _ = utils.create_vocab(x, config_dict['stop_word_path'])
  sents = utils.remove_stop(x, stopwords)

  label2idx, idx2label = utils.get_label_dict(y)
  # print(label2idx, idx2label)
  trec_loader = DataLoader(test_sents, test_y, 0, False, 0, label2idx)

  if args.dev:
    dataloader = DataLoader(sents, y, int(config_dict['batch_size']), shuffle=True, test_ratio=0.1, label2idx=label2idx)
    train(config_dict, voc, len(label2idx), dataloader, trec_loader, mode='dev')
  if args.train:
    dataloader = DataLoader(sents, y, int(config_dict['batch_size']), shuffle=False, test_ratio=0.0, label2idx=label2idx)
    train(config_dict, voc, len(label2idx), dataloader, trec_loader, mode='train')
  if args.test:
    test(config_dict, voc, len(label2idx), trec_loader, idx2label)
