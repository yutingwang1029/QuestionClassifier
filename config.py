import configparser

def get_config(path):
  config = configparser.ConfigParser()
  config.read(path)
  config_dict = {}

  for section in config.sections():
    for key in config[section].items():
      if section == 'MODEL':
        config_dict[key[0]] = int(key[1]) if key[1].isnumeric() else key[1] 
      else:
        config_dict[key[0]] = key[1]
    
  return config_dict