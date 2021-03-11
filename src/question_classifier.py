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
  parser.add_argument('--search', action='store_true', help="searching for hyper-params")
  parser.add_argument('--config', type=str, default=global_config_path, help="configuration path")
  args = parser.parse_args()
  return args
  
def test_trec(model, trec_loader):
  test_x, test_y = trec_loader.get_all()
  ret, y_preds, y_real = train_val(model, test_x, test_y)
  return ret, y_preds, y_real

def train_val(model, test_x, test_y):
  y_preds = list()
  y_real = list()
  with torch.no_grad():
    for j in range(len(test_x)):
      predict = model([test_x[j]])
      y_preds.extend(predict.argmax(dim=1).numpy().tolist())
      y_real.extend([test_y[j]])
  return np.sum(np.array(y_preds)==y_real)/len(y_real), y_preds, y_real

def train(config, voc, label_num, dataloader, trec_loader, save_path='', mode='dev'):
  model = QuestionClassifier(
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
    nn_hidden_dim_2=config['nn_hidden_dim_2'],
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
    batches = dataloader.get_length() // int(config['batch_size'])
    for _j in range(batches):
      features, labels = dataloader.next_batch()
      labels = torch.LongTensor(labels)
      optimizer.zero_grad()
      probs = model(features)
      loss = criterion(probs, labels)
      loss.backward()
      optimizer.step()
      if mode == 'dev':
        new_val_acc, _, _ = train_val(model, test_x, test_y)
        if new_val_acc > best_val_acc:
          early_stopping = 0
          best_val_acc = new_val_acc
        if early_stopping >= int(config['early_stopping']):
          print(f'dev training early stopping at epoch {i+1}')
          return model
    if i >= 3:
      scheduler.step()
    print(f"----- epoch {i+1} -----")
    if mode == 'dev':
      acc, _, _ = train_val(model, test_x, test_y)
      acc_trec, _, _ = test_trec(model, trec_loader)
      best_trec_acc = acc_trec if acc_trec > best_trec_acc else best_trec_acc
      print(f"epoch {i+1} finished, validation acc: {acc}, TREC 10 acc: {acc_trec}")
    else:
      acc_trec, _, _ = test_trec(model, trec_loader)
      if best_trec_acc < acc_trec:
        best_trec_acc = acc_trec
        best_model = copy.deepcopy(model.state_dict())
      print(f"epoch {i+1} finished, TREC 10 acc: {acc_trec}, best TREC 10 acc: {best_trec_acc} lr: {scheduler.get_last_lr()[0]}")
  if mode == 'train' and save_path != '':
    torch.save(best_model, save_path)
  return best_model, best_trec_acc

def cross_product(arr1, arr2):
  ret = []
  for item1 in arr1:
    for item2 in arr2:
      if isinstance(item1, list):
        temp = [ele for ele in item1]
      else:
        temp = [item1]
      temp.append(item2)
      ret.append(temp)
  return ret

def grid_search(config, voc, label_num, dataloader, trec_loader):
  lrs = [0.01, 0.011, 0.0115, 0.012]
  # bilstm_hiddens = [5, 10, 15, 20, 25, 30]
  # nn_input_dims = [50, 80, 100, 150, 200]
  nn_hidden_dims = [50, 80, 100, 150, 200]
  nn_hidden_dim2s = [50, 80, 100, 150, 200]
  # nn_hidden_dims = [10 * i for i in range(1, 20)]
  # t3 = nn_hidden_dims
  # lrs = [0.01]
  # bilstm_hiddens = [5]
  # nn_input_dims = [50]
  # nn_hidden_dims = [40,50]
  t1 = lrs
  # t1 = cross_product(lrs, bilstm_hiddens)
  t2 = cross_product(t1, nn_hidden_dims)
  t3 = cross_product(t2, nn_hidden_dim2s)

  result = []
  for setting in t3:
    new_config = copy.deepcopy(config)
    new_config['lr'] = setting[0]
    # new_config['bilstm_hidden_dim'] = setting[1]
    # new_config['nn_input_dim'] = setting[2]
    # new_config['nn_hidden_dim_1'] = setting[3]
    new_config['nn_hidden_dim_1'] = setting[1]
    new_config['nn_hidden_dim_2'] = setting[2]
    # result.append([s for s in setting])
    result.append([s for s in setting])

    new_dataloader = copy.deepcopy(dataloader)
    new_trecloader = copy.deepcopy(trec_loader)
    _, trec_acc = train(new_config, voc, label_num, new_dataloader, new_trecloader, mode='train')
    result[-1].append(trec_acc)
    print(result)
    title = 'lr,bilstm_hidden,nn_input_dim,nn_hidden_dim,acc\n'
    with open('./temp1.csv', 'w') as f:
      string = ''
      for line in result:
        new_string = ','.join(list(map(lambda x: str(x), line)))
        string += new_string + '\n'
      f.write(title+string)

def test(config, voc, label_num, trec_loader, idx2label):
  model = QuestionClassifier(
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
  if int(config['ensemble_size']) == 1:
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    trec_acc, final_preds, _ = test_trec(model, trec_loader)
  else:
    y_preds = []
    y_real = []
    final_preds = []
    weight = []
    for i in range(config['ensemble_size']):
      model.load_state_dict(torch.load(config[f'model_path_{i+1}']))
      model.eval()
      trec_acc, y_pred_temp, y_real = test_trec(model, trec_loader)
      for idx in range(len(y_pred_temp)):
        if len(y_preds) <= idx:
          y_preds.append([y_pred_temp[idx]])
          weight.append([trec_acc])
        else:
          y_preds[idx].append(y_pred_temp[idx])
          weight[idx].append(trec_acc)
    y_real = np.array(y_real)
    for idx in range(len(y_preds)):
      votes = dict([[i, 0] for i in range(len(y_real))])
      for j in range(len(y_preds[idx])):
        votes[y_preds[idx][j]] += weight[idx][j]
      final_preds.append(list(sorted(votes.items(), key=lambda x: x[1], reverse=True))[0][0])
    trec_acc = np.sum(np.array(final_preds) == y_real) / len(y_real)

  print(trec_acc)
  results = []
  for y in final_preds:
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
  trec_loader = DataLoader(test_sents, test_y, 0, False, 0, label2idx)

  if args.dev:
    dataloader = DataLoader(sents, y, int(config_dict['batch_size']), shuffle=True, test_ratio=0.1, label2idx=label2idx)
    train(config_dict, voc, len(label2idx), dataloader, trec_loader, mode='dev')
  if args.train:
    dataloader = DataLoader(sents, y, int(config_dict['batch_size']), shuffle=False, test_ratio=0.0, label2idx=label2idx)
    if int(config_dict['ensemble_size']) == 1:
      train(config_dict, voc, len(label2idx), dataloader, trec_loader, save_path=config_dict['model_path'], mode='train')
    else:
      for i in range(int(config_dict['ensemble_size'])):
        train(config_dict, voc, len(label2idx), dataloader, trec_loader, save_path=config_dict[f'model_path_{i+1}'], mode='train')
  if args.test:
    test(config_dict, voc, len(label2idx), trec_loader, idx2label)
  if args.search:
    dataloader = DataLoader(sents, y, int(config_dict['batch_size']), shuffle=False, test_ratio=0.0, label2idx=label2idx)
    grid_search(config_dict, voc, len(label2idx), dataloader, trec_loader)
