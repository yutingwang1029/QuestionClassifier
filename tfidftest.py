from utils import preprocessing, get_stopword, remove_stop, create_vocab
import configparser
import torch
import numpy as np
import wordEmbed
import math
from collections import defaultdict
#import word2sen

global_config_path = './config.ini'
# 计算TF(word代表被计算的单词，word_dict是被计算单词所在句子分词统计词频后的字典)
def tf(eveword, word_dict):
    return word_dict[eveword] / sum(word_dict.values())

# 统计含有该单词的句子数
def count_sentence(word, word_count):
    return sum([1 for i in word_count if i.get(word)])  # i[word] >= 1

# 计算IDF
def idf(word, word_count):
    return math.log(len(word_count) / (count_sentence(word, word_count) + 1))

# 计算TF-IDF
def tfidf(word, word_dict, word_count):
    return tf(word, word_dict) * idf(word, word_count)

def test_utils():
  config = configparser.ConfigParser()
  config.read(global_config_path)
  stopword_path = config["GENERAL"]["stop_word_path"]
  datapath = config["GENERAL"]["test_path"]
  x, y = preprocessing(datapath)
  stopword_list = get_stopword(stopword_path)
  sents = remove_stop(x, stopword_list)
  vo = create_vocab(sents)
  #print(vo)
  vecs = []
  for sent in sents:
      vec = []
      for token in sent:
        vec.append(vo[token])
      vecs.append(vec)
  #print(vecs[0])
  randomVec = wordEmbed.RandomWordVec()
  senvec=[]
  for i in range(len(vecs)):
    input = torch.LongTensor(vecs[i])
    tensor=randomVec.forward(input)
    senvec.append(tensor)
    #print(len(tensor))
    #print(sum(tensor)/(len(tensor)))
  #print(senvec[0])
  #print((senvec[0])[0])

  #下面求每一个单词的tfidf
  word_count = []
  
  for sentence in sents:
    word_dict = defaultdict(int)
    for word in sentence:
        word_dict[word] += 1
    word_count.append(word_dict)
  print(word_count)
    #print(word_dict['what'])
  for eveword in word_dict:
     #print(tf(eveword,word_dict))
      print(count_sentence(eveword,word_count))
      print(idf(eveword,word_count))
      #print(tfidf(eveword, word_dict, word_count))
    

  
    
    
test_utils()