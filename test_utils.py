from utils import preprocessing, get_stopword, remove_stop
import configparser

global_config_path = './config.ini'

def test_utils():
  config = configparser.ConfigParser()
  config.read(global_config_path)
  stopword_path = config["GENERAL"]["stop_word_path"]
  datapath = config["GENERAL"]["data_path"]
  x, y = preprocessing(datapath)
  stopword_list = get_stopword(stopword_path)
  sents = remove_stop(x, stopword_list)
  print(sents)

test_utils()