#import numpy as np
#import time
#import torch.nn.functional as F
#import pickle
#import json
#from pathlib import Path
#import os
#from src.envs.envList import *

#import numpy as np
import torch
#import pybullet as p

#from src.envs.env_params import get_env_params
#from src.envs.color_generation import infer_color
#from extra_utils.data_utils import get_obs_cont, fix_quaternions, one_hot, get_tokens
#from create_simple_dataset import has_concrete_object_ann, check_if_exact_one_object_from_obs, get_new_obs_from_obs
#from src.envs.utils import save_traj
#import uuid
#from src.envs.reward_function import get_reward_from_state, sample_descriptions_from_state
#
#from constants import *
#from extra_utils import distribute_tasks
#import os

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)
