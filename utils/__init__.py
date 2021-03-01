from .data_preprocess import *

from config import data_path

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
