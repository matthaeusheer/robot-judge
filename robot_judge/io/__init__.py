import os
from glob import glob
import random

from common import DATA_DIR_PATH
RANDOM_SEED = 42


def read_file(file_name):
    """Reads all lines of a text file and concatenates them with whitespaces."""
    with open(file_name, 'r') as infile:
        return ' '.join(infile.readlines())


class ProblemSet1Io:
    """Data loading functionality for problem set 1."""
    def __init__(self, data_dir='assignment_1'):
        self.data_dir = data_dir

    @property
    def get_cases_file_paths(self):
        cases_file_names = glob(os.path.join(DATA_DIR_PATH, self.data_dir, '[0-9]' * 4 + '_*.txt'))
        assert len(cases_file_names) != 0, 'Something went wrong. Did you place the data correctly into data dir?'
        return cases_file_names

    def read_multiple_cases_files(self, n_samples=-1):
        cases_file_paths = self.get_cases_file_paths
        if n_samples != -1:
            random.seed(RANDOM_SEED)
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