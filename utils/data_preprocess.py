import utils
import re
from collections import Counter

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

# creating list of stop words from stopwords.txt
def get_stopword(datapath):
    stopwords = []
    with open(datapath) as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords

#function to remove stop words
def remove_stop(sent_list, stopwords):
    # for removing stop words from dictionary list
    sents = []
    for sent in sent_list:
        list_without_stop = [word for word in sent if not word in stopwords]
        sents.append(list_without_stop)
    return sents

def create_vocab(sents, stopword_path):
  stopword_list = get_stopword(stopword_path)
  sent_list = remove_stop(sents, stopword_list)
  word_dict = dict()
  for sent in sent_list:
    for token in sent:
      if token not in word_dict:
        word_dict[token] = 1
      else:
        word_dict[token] += 1
  sort_word_dict = dict(sorted(word_dict.items(), key=lambda item: item[1]))
  word_idx_dict = dict()
  idx = 0
  for item in sort_word_dict:
    word_idx_dict[item] = idx
    idx += 1
  return word_idx_dict, sent_list

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
    token_of_sent = [word.lower() for word in token_of_sent]
    for idx in range(len(token_of_sent)):
        if token_of_sent[idx] == 'u' and idx < len(token_of_sent) - 1:
            if token_of_sent[idx+1] == 's':
                token_of_sent[idx] = 'u.s.'
        elif token_of_sent[idx] == 'e' and idx < len(token_of_sent) - 1:
            if token_of_sent[idx+1] == 'mail' or token_of_sent[idx+1] == 'mails':
                token_of_sent[idx] = 'email'
                token_of_sent[idx+1] = 's'
    tokens.append(token_of_sent)
  return tokens

