import re

def load_data(path):
  data = []
  with open(path, 'r') as f:
    for line in f:
      temp = line.split(' ', 1)
      label, sent = temp[0], temp[1]
      data.append((label, sent))
  return data

def load_labels(path):
  with open(path, 'r') as f:
    return f.read().split('\n')

def load_stop_words(path):
  with open(path, 'r') as f:
    return f.read().split()

def load_pre_train(path):
  pre_train_words = {}
  with open(path, 'r') as f:
    for line in f:
      line = re.sub(r'\s+',' ',line).rstrip()
      pair = line.split(' ')
      key = pair[0]
      value = [float(x) for x in pair[1:]]
      pre_train_words[key] = value
      pre_train_words[key.lower()] = value
    return pre_train_words

def load_vocabulary(path):
  vocabulary = ['#unk#']
  with open(path, 'r') as f:
    for line in f:
      temp = line.split()
      vocabulary.append(temp[0])
  return vocabulary

def create_word_embedding(pretrain_words_dict, vocabulary):
  unk_embedding = pretrain_words_dict['#unk#']
  pretrain_weight = []
  for word in vocabulary:
    if word in pretrain_words_dict:
      pretrain_weight.append(pretrain_words_dict[word])
    else:
      pretrain_weight.append(unk_embedding)
  return pretrain_weight