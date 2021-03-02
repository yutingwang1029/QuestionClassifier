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
  return
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

# creating list of stop words from Stop_words.txt
def get_stopword(datapath):
    STOP_WORDS = []
    with open(datapath) as f:
        for line in f:
            STOP_WORDS.append(line.strip())
    return STOP_WORDS

#function to remove stop words
def remove_stop(sent_list, STOP_WORDS):
    # for removing stop words from dictionary list
    sents = []
    for sent in sent_list:
        list_without_stop = [word for word in sent if not word in STOP_WORDS]
        sents.append(list_without_stop)
    return sents

def create_vocab(sent_list):
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
    return word_idx_dict

def create_word_enbedding(encoded_sentences):
    word_vectors = [i for i in range(len(encoded_sentences))]
    emb_dim = 3
    for i in range(len(encoded_sentences)):
        emb_layer = nn.Embedding(len(encoded_sentences[i]), emb_dim)
        word_vectors[i] = emb_layer(torch.LongTensor(encoded_sentences[i]))
    return word_vectors

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
    tokens.append(token_of_sent)
  return tokens

