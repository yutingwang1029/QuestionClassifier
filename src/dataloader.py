import random
from utils import get_label_vecs, get_tags

random.seed(3)

class DataLoader:
  def __init__(self, x, y, batch_size, shuffle=True, test_ratio=0.1, label2idx={}):
    temp = list(zip(x, y))
    if shuffle:
      random.shuffle(temp)
    nx, ny = zip(*temp)
    nx = list(nx)
    ny = list(ny)
    
    self.total_length = len(nx)
    self.test_length = int(self.total_length * test_ratio)

    self.test_x = nx[self.total_length-self.test_length:]
    self.test_y = ny[self.total_length-self.test_length:]
    self.train_x = nx[:self.total_length-self.test_length]
    self.train_y = ny[:self.total_length-self.test_length]

    self.length = len(self.train_x)
    
    self.label2idx = label2idx
    self.batch_size = batch_size
    self.pointer = 0
  
  def get_test_data(self):
    return self.test_x, get_tags(self.test_y, self.label2idx)
  def get_length(self):
    return 2*self.length
  def next_batch(self):
    old_pointer = self.pointer
    if self.pointer + self.batch_size < self.length:
      self.pointer += self.batch_size
      return \
        self.train_x[old_pointer:self.pointer], \
        get_tags(self.train_y[old_pointer:self.pointer], self.label2idx)
    else:
      self.pointer = (self.pointer + self.batch_size) % self.length
      return self.train_x[old_pointer:], \
        get_tags(self.train_y[old_pointer:], self.label2idx)
  
  def get_all(self):
    return self.train_x, get_tags(self.train_y, self.label2idx)
