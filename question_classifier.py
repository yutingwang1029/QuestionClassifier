import configparser
import torch.optim as optim
from model import QuestionClassifier
import torch.nn as nn

global_config_path = './config.ini'

def cmdparser():
  # todo: parse the cmd argument
  pass

def train(config, voc, label_num, training_data):
  model = QuestionClassifier(
    bow=False,
    voc=voc,
    pretrain_embedding_path='',
    freeze=False,
    random_or_word2vec='random',
    bilstm_input_dim=3,
    bilstm_hidden_dim=20,
    bilstm_max_len=11,
    nn_input_dim=3,
    nn_hidden_dim=100,
    nn_output_size=label_num
  )
  optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-5)
  criterion = nn.CrossEntropyLoss()
  for i in range(config.epoches):
    print(f"----- epoch {i} -----")
    batches = training_data.length / config.batch_size
    for _j in range(batches):
      features, labels = training_data.next_batch()
      model.zero_grad()
      probs = model(features)
      loss = criterion(probs, labels)
      loss.backward()
      optimizer.step()
    


def test():
  pass

if __name__ == "__main__":
  config = configparser.ConfigParser()
  config.read(global_config_path)
  datapath = config["GENERAL"]["data_path"]
  # x, _ = preprocessing(datapath)