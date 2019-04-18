import os
from robot_judge import utils
import pickle

from common import MODELS_DIR_PATH


def store_model(model, dir_path=MODELS_DIR_PATH, file_name=None):
    """Pickle a given model with an optional file_name into dir_path, which is the data/models directory by default."""

    if file_name is None:
        file_name = 'unknown'

    file_path = os.path.join(dir_path, file_name)
    file_path += ('_' + utils.get_datetime_tag() + 'pkl')

    with open(file_path, 'wb') as outfile:
        pickle.dump(model, outfile)

    print(f'Pickled model {file_name} to {dir_path}')


def load_model(dir_path=MODELS_DIR_PATH, file_name=''):
    """Unpickle a model from file system."""

    with open(os.path.join(dir_path, file_name), 'rb') as infile:
        model = pickle.load(infile)

    print(f'Un-pickled model {file_name} from {dir_path}')

    return model
