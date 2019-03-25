"""
This script let's us process a whole corpus dictionary (labelled texts) on Euler.
"""

from robot_judge.utils.timer import Stopwatch
from robot_judge.io import ProblemSet1Io
from robot_judge.exploration.corpus_analysis import process_corpus_dict_to_doc_dict

N_SAMPLED_CASES = 50  # How many cases should be sampled to work on.
DATA_DIR_NAME = 'assignment_1'

timer = Stopwatch()
timer.start()

io = ProblemSet1Io(data_dir=DATA_DIR_NAME)
corpus_dict = io.read_multiple_cases_files(n_samples=N_SAMPLED_CASES, random_seed=1)
timer.stop()

print('Files loaded in {} secs.'.format(timer.total_run_time))

timer.start()
_ = process_corpus_dict_to_doc_dict(corpus_dict, store=True, n_threads=12)
timer.stop()

print('Finished processing in {} secs.'.format(timer.total_run_time))