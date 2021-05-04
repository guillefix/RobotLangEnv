import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

from src.envs.environments import playEnv

class UR5PlayAbsRPY1Obj(playEnv):
	def __init__(self, num_objects = 3, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         # obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         obj_lower_bound = [-0.4, 0.01, 0.03], obj_upper_bound = [0.4, 0.3, 0.03], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_rpy', show_goal=False, arm_type='UR5')


# table_ranges = [(-0.55, 0.55), (0., 0.35)]
