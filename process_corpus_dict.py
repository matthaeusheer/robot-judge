"""
This script let's us process a whole corpus dictionary (labelled texts) on Euler.
"""

from robot_judge.io import DataLoader
from robot_judge.exploration.corpus_analysis import process_corpus_dict_to_doc_dict

N_SAMPLED_CASES = 5000  # How many cases should be sampled to work on.
DATA_DIR_NAME = 'assignment_1'  # Where to fetch raw data
DATA_DIR_NAME_OUTPUT = 'assignment_1_processed_docs'  # Where to store processed data

io = DataLoader(data_dir=DATA_DIR_NAME)
corpus_dict = io.read_multiple_cases_files(n_samples=N_SAMPLED_CASES, random_seed=1)

_ = process_corpus_dict_to_doc_dict(corpus_dict, store=True, n_threads=12,
                                    data_sub_dir=DATA_DIR_NAME_OUTPUT)