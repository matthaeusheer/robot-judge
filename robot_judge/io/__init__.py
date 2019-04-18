import os
from glob import glob
import random
import pandas as pd

from common import DATA_DIR_PATH
RANDOM_SEED = 42


def read_file(file_name):
    """Reads all lines of a text file and concatenates them with whitespaces."""
    with open(file_name, 'r') as infile:
        return ' '.join(infile.readlines())


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


class DataLoader:
    """Data loading functionality for problem set 1."""
    def __init__(self, data_dir='assignment_data', metadata_file='case_metadata.csv'):
        self.data_dir = data_dir
        self.metadata_file = metadata_file

    @property
    def get_cases_file_paths(self):
        cases_file_names = glob(os.path.join(DATA_DIR_PATH, self.data_dir, '[0-9]' * 4 + '_*.txt'))
        assert len(cases_file_names) != 0, 'Something went wrong. Did you place the data correctly into data dir?'
        return cases_file_names

    def read_multiple_cases_files(self, n_samples=-1, random_seed=None):
        """Samples a number of n_samples case files from the data folder."""
        cases_file_paths = self.get_cases_file_paths
        if n_samples != -1:
            random.seed(random_seed if random_seed else RANDOM_SEED)
            cases_file_paths = random.sample(cases_file_paths, n_samples)

        cases_dict = {}
        for case in cases_file_paths:
            base_path, file_name = os.path.split(case)
            cases_dict[file_name.split('.')[0]] = read_file(os.path.join(base_path, file_name))

        return cases_dict

    @staticmethod
    def get_year_from_case_title(case_title):
        year = case_title.split('_')[0]
        assert len(year) == 4, 'Year must be 4 digits long.'
        return int(year)

    def get_target_values_df(self):
        """Loads the bare target values into a data frame."""
        return pd.read_csv(os.path.join(DATA_DIR_PATH, self.data_dir, self.metadata_file), index_col='caseid')

    def get_target_values(self, target_labels, target):
        """For a given list of target labels, return the corresponding target values. The target_labels list
        would hold case labels while the target_values list holds the case reversed / not reversed integer."""

        target_df = pd.read_csv(os.path.join(DATA_DIR_PATH, self.data_dir, self.metadata_file), index_col='caseid')
        all_case_targets = target_df.to_dict()[target]

        target_values = []
        for label in target_labels:
            target_values.append(all_case_targets[label])

        return target_values
