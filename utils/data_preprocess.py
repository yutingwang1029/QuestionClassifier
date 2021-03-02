import utils
import re

# preprocessing the data

def preprocessing(datapath):
  """
  description:
    preprocess the raw data
  param:
    datapath: path to raw data file
  return:
    (x, y): x be list of the tokens list and y be the corresponding label of each token list
  """
  tempx, y = split_label_sent(datapath)
  x = tokenization(tempx)
  return x, y

def split_label_sent(datapath):
  """
  description:
    split the labels and sentences
  param:
    datapath: path to raw data file
  return:
    (x, y): x be the sentences and y be the labels
  """
  with open(datapath, 'r') as f:
    raw_string = f.read()
    lines = raw_string.split('\n')
    y = list(map(lambda line: line.split(' ')[0], lines))
    x = [lines[i][len(y[i])+1:] for i in range(len(lines))]
    return x, y

def tokenization(sents):
  """
  description:
    tokenize the sentences
  params:
    sents: list of sentences
  return:
    tokens: 2-D array, representing the token list for each sentence
  """
  tokens = []
  for sent in sents:
    token_of_sent = re.findall(r"[\w']+", sent)
    # todo: more rules to more accurately tokenize the sentences
    tokens.append(token_of_sent)
  return tokens

