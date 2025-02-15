from run_inference import run
from extra_utils import distribute_tasks
import os
import argparse
parser = argparse.ArgumentParser(description='Evaluate LangGoalRobot environment')
#parser.add_argument('--eval_train_demos', action='store_true', help='whether to trained_demos')
parser.add_argument('--base_filenames_file', help='file listing demo sequence ids')
parser.add_argument('--sample_goals', action='store_true', help='whether to sample random goals for each task')
parser.add_argument('--num_tasks', type=int, default=1, help='number of tasks (overriden by number of sequence ids if base_filenames_file is not None)')
parser.add_argument('--num_repeats', type=int, default=1, help='number of times each demo should be used')
parser.add_argument('--using_model', action='store_true', help='whether to evaluate a model or to evaluate a recorded trajectory')
# parser.add_argument('--predict_ttg', action='store_true', help='whether the model predicts time to go')
parser.add_argument('--simple_obs', action='store_true', help='whether to use the simple_obs in the env')
parser.add_argument('--computing_loss', action='store_true', help='whether to compute the loss of a recorded trajectory')
parser.add_argument('--computing_relabelled_logPs', action='store_true', help='whether to compute the logP of the trajectory chunk before a state where some goal(s) have been completed')
parser.add_argument('--save_eval_results', action='store_true', help='whether to save evaluation results')
parser.add_argument('--save_relabelled_trajs', action='store_true', help='whether to save the relabelled subtrajectories')
parser.add_argument('--render', action='store_true', help='whether to render the environment')
parser.add_argument('--goal_str', help='specify goal string (if not specified, we use the one from the demo)')
parser.add_argument('--zero_seed', action='store_true', help='whether to seed the obs and acts with zeros or with the beginning of the demo')
parser.add_argument('--random_seed', action='store_true', help='whether to seed the obs and acts with a standard normal distribution')
parser.add_argument('--using_torchscript', action='store_true', help='whether to use torchscript compiled model or not')
parser.add_argument('--session_id', help='the session from which to restore the demo')
parser.add_argument('--rec_id', help='the recording from within the session to retrieve as demo')
parser.add_argument('--pretrained_name', default=None, help='experiment name to retrieve the model from (if evaling a model)')
parser.add_argument('--experiment_name', default="default", help='experiment name to save results')
parser.add_argument('--varying_args', default="session_id,rec_id", help='comma-separated list of arguments that vary in the experiment')
parser.add_argument('--restore_objects', action='store_true', help='whether to restore the objects as they were in the demo')
parser.add_argument('--temp', type=float, default=1.0, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')
parser.add_argument('--dynamic_temp', action='store_true', help='whether to use the dynamic temperature trick to encourage exploration')
parser.add_argument('--dynamic_temp_delta', type=float, default=0.99, help='the decay/smoothing parameter in the dynamic temp trick algorithm')
parser.add_argument('--save_chunk_size', type=int, default=120, help='the number of frames previous to achievement of a goal to consider part of the episode saved for that goal')
parser.add_argument('--max_number_steps', type=int, default=3000, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')
parser.add_argument('--times_to_go_start', type=int, default=200, help='the initial times to go when set manually')
parser.add_argument('--version_index', type=int, default=-1, help='the version of the checkpoint to load')
parser.add_argument('--n_output_samples', type=int, default=1, help='the number of samples of outputs of the model, useful for using with ttg')

args = parser.parse_args()

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

#TODO: add task where we sample a random instruction

tasks = []

if "DATA_FOLDER" not in os.environ:
    data_folder="/home/guillefix/code/inria/UR5/"
else:
    data_folder = os.environ["DATA_FOLDER"]

common_args = vars(args).copy()
del common_args["base_filenames_file"]
del common_args["num_repeats"]
del common_args["sample_goals"]
del common_args["num_tasks"]
if args.base_filenames_file is not None:
    with open(args.base_filenames_file, "r") as f:
        filenames = [x[:-1] for x in f.readlines()] # to remove new lines
    num_tasks = len(filenames)
    tasks = args.num_repeats*list(map(lambda x: {**common_args, "session_id": x.split("_")[1], "rec_id": x.split("_")[5]}, filenames))
elif args.sample_goals:
    from extra_utils.run_utils import generate_goal
    tasks = args.num_repeats*[{**common_args, "goal_str": generate_goal()} for i in range(args.num_tasks)]

    #filenames = filenames[:2]

#common_args = {"restore_objects": True}
tasks = distribute_tasks(tasks, rank, size)
#print(tasks)
for task in tasks:
    run(**task)
print("Finished. Rank: "+str(rank))
