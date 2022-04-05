from evaluate import evaluate
from extra_utils import distribute_tasks
import os
import argparse
parser = argparse.ArgumentParser(description='Evaluate LangGoalRobot environment')
parser.add_argument('--eval_train_demos', action='store_true', help='whether to trained_demos')
parser.add_argument('--base_filenames_file', help='file listing demo sequence ids')
args = parser.parse_args()

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

tasks = []

if "DATA_FOLDER" not in os.environ:
    data_folder="/home/guillefix/code/inria/UR5/"
else:
    data_folder = os.environ["DATA_FOLDER"]

if args.base_filenames is not None:
    with open(args.base_filenames_file, "r") as f:
        filenames = [x[:-1] for x in f.readlines()] # to remove new lines
    #filenames = filenames[:2]

if args.eval_train_demos:
    tasks = list(map(lambda x: {"session_id": x.split("_")[1], "rec_id": x.split("_")[5], "restore_objects": True}, filenames))
    tasks = distribute_tasks(tasks, rank, size)

else:
    tasks = list(map(lambda x: {"session_id": x.split("_")[1], "rec_id": x.split("_")[5], "restore_objects": False}, filenames))
    tasks = distribute_tasks(tasks, rank, size)

for task in tasks:
    evaluate(**task)
