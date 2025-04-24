import yaml
import pickle
import os
import logging

def load_nested_yaml(file_path):
    """
    Loads and returns nested YAML configuration data from a given file path.
    
    Args:
        file_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: A dictionary representing the YAML configuration.
        
    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")
    
    return config

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
  """
  Sets up a logger with a stream handler and a formatter.

  Args:
    name: The name of the logger.
    log_level: The logging level. Defaults to logging.INFO.

  Returns:
    A configured logging.Logger object.
  """
  logger = logging.getLogger(name)
  if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  logger.setLevel(log_level)
  return logger


def load_pickle(pickle_path):
    """ Load a pickle file

    Args:
        pickle_path (str):

    Returns:
        pickle file content
    """
    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def write_pickle(pickle_path, data):
    """ Write data in a pickle path

    Args:
        pickle_path (str):
        data:

    Returns:

    """
    with open(pickle_path, 'wb') as handle:
        pickle.dump(data, handle)