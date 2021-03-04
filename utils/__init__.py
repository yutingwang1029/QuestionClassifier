from .data_preprocess import *

def materialize(string, dst):
  """
  description:
    temporarily materialize some intermediate result.
    a tool function for testing
  params:
    string: the text to be materialzed
    dst: the destination of the intermediate file
  """
  with open(dst, 'w') as f:
    f.write(string)

def word2vec(vo, sents):
  vecs = []
  for sent in sents:
    vec = []
    for word in sent:
      vec.append(vo[word])
    vecs.append(vec)
  return vecs