from utils import preprocessing
import configparser

config_path = './config.ini'

def cmdparser():
  # todo: parse the cmd argument
  pass

def train():
  # todo: train the model
  pass

def test():
  pass

if __name__ == "__main__":
  config = configparser.ConfigParser()
  config.read(config_path)
  datapath = config["GENERAL"]["data_path"]
  x, _ = preprocessing(datapath)