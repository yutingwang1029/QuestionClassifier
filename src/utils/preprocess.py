import numpy as np
import re

def create_training_data(raw_data_path, training_data_path, dev_data_path):
  with open(raw_data_path,'r') as f:
    raw_data = f.read().split('\n')

  # Leave one out policy, 10% for validation
  rand_idx = np.random.randint(0, len(raw_data), int(len(raw_data) * 0.1))
  dev = [raw_data[i] for i in rand_idx]
  train = [s for s in raw_data if s not in dev]

  with open(training_data_path, 'w') as f:
    for item in train:
      f.write(f'{item}\n')

  with open(dev_data_path, 'w') as filehandle:
    for item in dev:
      filehandle.write(f'{item}\n')
  return raw_data

def collect_labels(label_file_path, raw_data):
  label_set = dict()
  for item in raw_data:
    pattern = re.compile(r'\w+:\w+\s')
    label = pattern.search(item).group().strip()
    if label not in label_set:
      label_set[label] = 1
  labels = [item[0] for item in label_set.items()]
  sorted(labels)
  with open(label_file_path, 'w') as f:
    for label in labels:
      f.write(f'{label}\n')

def get_stop_words(stop_word_path):
  with open(stop_word_path, 'r') as f:
    return f.read().split()

def build_vocabulary(raw_data, vocabulary_path, stop_words):
  features = [re.sub(r'\w+:\w+\s', '', sent).lower() for sent in raw_data]
  # tokenization
  words = ' '.join(features).split()
  vocabulary = {}
  for word in words:
    if word in stop_words:
      continue
    if word in vocabulary.keys():
      vocabulary[word] += 1
    else:
      vocabulary[word] = 1

  vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))

  # word_count = sum(vocabulary.values())
  voc = [(f"{key} {val}") for key, val in vocabulary.items()]

  with open(vocabulary_path, 'w') as f:
    for word_idx in voc:
      f.write(f'{word_idx}\n')

def preprocess(
    raw_data_path, \
    training_data_path, \
    dev_data_path, \
    label_file_path, \
    stop_word_path, \
    vocabulary_path
  ):
  raw_data = create_training_data(
    raw_data_path,
    training_data_path,
    dev_data_path
  )
  collect_labels(label_file_path, raw_data)
  stop_words = get_stop_words(stop_word_path)
  build_vocabulary(raw_data, vocabulary_path, stop_words)