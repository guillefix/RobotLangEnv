import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

from src.envs.environments import playEnv
from src.envs.descriptions import generate_all_descriptions
from src.envs.env_params import get_env_params

from extra_utils.run_utils import *
from constants import *
import gym.spaces as spaces
import pickle
from src.envs.reward_function import get_reward_from_state, sample_descriptions_from_state

class UR5PlayAbsRPY1Obj(playEnv):
	def __init__(self, num_objects = 3, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True):
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.4, 0.01, 0.03], obj_upper_bound = [0.4, 0.3, 0.03], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_rpy', show_goal=False, arm_type='UR5')


SUCCESS_REWARD = 1.0


class ExtendedUR5PlayAbsRPY1Obj(UR5PlayAbsRPY1Obj):
	def __init__(self, obs_scaler = None, acts_scaler = None, prev_obs = None, prev_acts = None, times_to_go = None,
				 save_relabelled_trajs = False, check_completed_goals = True, sample_random_goal = False,
				 vocab_size = 73, max_episode_length = 5000,
				 desc_max_len = 11, obs_mod="",
				 goal_str = None, use_dict_space = False,
				 args = {},
				 num_objects = 3, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True):
		super().__init__(num_objects, env_range_low, env_range_high, goal_range_low, goal_range_high, use_orientation)

		high = np.array([6, 6, 6, 1, 1, 1, 1, 1])
		self.action_space = spaces.Box(-high, high)
		acts_range = 3*np.ones(8)
		self.use_dict_space = use_dict_space
		self.sample_random_goal = sample_random_goal
		if use_dict_space:
			if prev_obs is not None and prev_acts is not None:
				if times_to_go is not None:
					self.observation_space = spaces.Dict({"times_to_go": spaces.Box(low=0, high=10000, shape=times_to_go.shape), "ann": spaces.MultiDiscrete(desc_max_len*[vocab_size]), "obs": spaces.Box(low=-6, high=6, shape=prev_obs.shape), "acts": spaces.Box(low=-6, high=6, shape=prev_acts.shape)})
				else:
					self.observation_space = spaces.Dict({"ann": spaces.MultiDiscrete(desc_max_len*[vocab_size]), "obs": spaces.Box(low=-6, high=6, shape=prev_obs.shape), "acts": spaces.Box(low=-6, high=6, shape=prev_acts.shape)})
			else:
			    self.observation_space = spaces.Dict({"times_to_go": spaces.Box(low=0, high=10000, shape=(1,)), "ann": spaces.MultiDiscrete(desc_max_len*[vocab_size]), "obs": spaces.Box(low=-6, high=6, shape=(1,)), "acts": spaces.Box(low=6, high=6, shape=(1,))})
		else:
			if prev_obs is not None and prev_acts is not None:
				if times_to_go is not None:
					self.observation_space = spaces.Tuple((spaces.Box(low=0, high=10000, shape=times_to_go.shape),spaces.MultiDiscrete(desc_max_len*[vocab_size]), spaces.Box(low=-6, high=6, shape=prev_obs.shape), spaces.Box(low=-6, high=6, shape=prev_acts.shape)))
				else:
					self.observation_space = spaces.Tuple((spaces.MultiDiscrete(desc_max_len*[vocab_size]), spaces.Box(low=-6, high=6, shape=prev_obs.shape), spaces.Box(low=-6, high=6, shape=prev_acts.shape)))
			else:
			    self.observation_space = spaces.Tuple((spaces.Box(low=0, high=10000, shape=(1,)),spaces.MultiDiscrete(desc_max_len*[vocab_size]), spaces.Box(low=-6, high=6, shape=(1,)), spaces.Box(low=6, high=6, shape=(1,))))

		object_types = pickle.load(open(root_folder+"object_types.pkl","rb"))
		self.goal_str = goal_str
		self.env_params['types'] = object_types
		self.obs_mod = obs_mod
		self.args = args
		self.obs_scaler = obs_scaler
		self.acts_scaler = acts_scaler
		self.desc_max_len = desc_max_len
		self.tokens = None
		self.t = 0
		self.max_episode_length = max_episode_length
		if prev_obs is not None and prev_acts is not None:
		    self.prev_obs, self.prev_acts = scale_inputs(obs_scaler, acts_scaler, prev_obs, prev_acts, "noarm" in obs_mod)
		else:
		    self.prev_obs = prev_obs
		    self.prev_acts = prev_acts
		self.times_to_go = times_to_go
		self.initial_state = None
		self.save_relabelled_trajs = save_relabelled_trajs
		self.check_completed_goals = check_completed_goals
		self.old_descriptions = []
		if self.save_relabelled_trajs:
		    self.obss = []
		    self.actss = []
		    self.joints = []
		    self.acts_rpy = []
		    self.acts_rpy_rel = []
		    self.velocities = []
		    self.targetJoints = []
		    self.gripper_proprioception = []

	def reset(self, o = None, vr =None, description=None, info_reset=None, joint_poses=None, objects=None, restore_objs=False):
		if self.sample_random_goal:
			self.goal_str = generate_goal("single" in self.obs_mod)
		if description is not None:
			self.goal_str = description
		self.old_descriptions = []
		if self.save_relabelled_trajs:
		    self.obss = []
		    self.actss = []
		    self.joints = []
		    self.acts_rpy = []
		    self.acts_rpy_rel = []
		    self.velocities = []
		    self.targetJoints = []
		    self.gripper_proprioception = []
		self.t = 0
		if not self.physics_client_active:
		    self.activate_physics_client(vr)
		    self.physics_client_active = True
		# print("env info_reset:", info_reset)
		self.instance.reset(o, info_reset=info_reset, description=self.goal_str, joint_poses=joint_poses, objects=objects, restore_objs=restore_objs)
		obj_stuff = self.instance.get_stuff_to_save()
		self.tokens = get_tokens(self.goal_str, max_length=self.desc_max_len, obj_stuff=obj_stuff)
		obj_index = -1
		if self.obs_mod in ["obs_cont_single_nocol_noarm_trim_scaled", "obs_cont_single_nocol_noarm_scaled"]:
			has_conc_obj, color, object_type = has_concrete_object_ann(self.goal_str)
			print(color, object_type)
			# assert has_conc_obj
			# exact_one_object, obj_index = check_if_exact_one_object_from_obs(obs_cont, disc_cond, color, object_type)
			objects = self.instance.objects_added
			matches = 0
			for i, obj in enumerate(objects):
			    if obj['type'] == object_type and obj['color'] == color:
			        matches += 1
			        obj_index = i
		self.obj_index = obj_index
		if self.times_to_go is not None:
			inputs = (self.tokens, self.prev_obs, self.prev_acts, self.times_to_go)
		else:
			inputs = (self.tokens, self.prev_obs, self.prev_acts)
		if self.use_dict_space:
			if self.times_to_go is not None:
				inputs = {"times_to_go": inputs[0], "ann": inputs[1], "obs": inputs[2], "acts": inputs[3]}
			else:
				inputs = {"ann": inputs[0], "obs": inputs[1], "acts": inputs[2]}
		# return inputs, 0, False, {}
		return inputs


	# Classic dict-env .step function
	def inner_step(self, action):
	    # action = np.clip(action, self.action_space.low, self.action_space.high)
	    targetPoses = self.instance.perform_action(action, self.action_type)
	    self.instance.runSimulation()
	    obs = self.instance.calc_state()
	    done = False
	    return obs['observation'], 0, done, {'target_poses': targetPoses}

	def update_traj_data1(self, action):
		action_quat = acts_scaler.inverse_transform(action)[0]
		action = scale_outputs(self.acts_scaler, action)
		state = self.instance.calc_state()
		# print(state)
		rel_xyz = np.array(act_pos)-np.array(state['observation'][0:3])
		rel_rpy = np.array(acts_euler) - np.array(p.getEulerFromQuaternion(state['observation'][3:7]))
		action_rpy_rel = np.array(list(rel_xyz)+list(rel_rpy)+act_gripper)
		self.actss.append(action_quat)
		self.obss.append(state['observation'])
		self.joints.append(state['joints'])
		self.acts_rpy.append(action)
		self.acts_rpy_rel.append(action_rpy_rel)
		self.velocities.append(state['velocity'])
		self.gripper_proprioception.append(state['gripper_proprioception'])

	def find_completed_goals(self, obs):
		obj_stuff = self.instance.get_stuff_to_save()
		# start_time = time.time()
		train_descriptions, test_descriptions = sample_descriptions_from_state(self.initial_state, obs, obj_stuff, env.instance.env_params)
		# print("--- Description computing time: %s seconds ---" % (time.time() - start_time)) #smol
		descriptions = train_descriptions + test_descriptions
		new_descriptions = []
		if descriptions != self.old_descriptions:
		    new_descriptions = [desc for desc in descriptions if desc not in old_descriptions]
		    lost_descriptions = [desc for desc in old_descriptions if desc not in descriptions]
		    self.old_descriptions = descriptions
		    # if > save_chunk_size:
		    print("New descriptions: "+", ".join(new_descriptions))
		    print("Lost descriptions: "+", ".join(lost_descriptions))

		if self.t>1 and self.save_relabelled_trajs:
			if len(new_descriptions)>0:
				obj_stuff = self.instance.get_stuff_to_save()
				traj_data = (self.actss, self.obss, self.joints, self.targetJoints, self.acts_rpy, self.acts_rpy_rel, self.velocities, self.gripper_proprioception)
				save_traj(new_descriptions, self.args, traj_data, obj_stuff)

		return new_descriptions

	def step(self, action):
		# update traj data
		if self.save_relabelled_trajs:
			self.update_traj_data1(action)

		action_scaled = action
		action = scale_outputs(self.acts_scaler, action)
		obs, r, done, info = self.inner_step(action)
		# obs[8:11] = [-0.3,0,max_size/2]
		# obs[8:11] = [-0.6,0,0.08]
		# obs[8:11] = [0.6,0.6,0.24]
		# obs[8:11] = [-0.3,0.4,0.04]
		# env.instance.reset_objects(obs)
		if self.save_relabelled_trajs:
		    self.targetJoints.append(info["target_poses"])
		inputs = make_inputs(self.obs_scaler, self.acts_scaler, obs, action_scaled, self.prev_obs, self.prev_acts, self.times_to_go, self.tokens, self.obj_index, self.obs_mod, convert_to_torch=False)
		if self.use_dict_space:
			if self.times_to_go is not None:
				inputs = {"times_to_go": inputs[0], "ann": inputs[1], "obs": inputs[2], "acts": inputs[3]}
			else:
				inputs = {"ann": inputs[0], "obs": inputs[1], "acts": inputs[2]}

		info = {**info, "raw_obs": obs}

		# check if succesful
		if self.initial_state is None:
			self.initial_state = obs
		obj_stuff = self.instance.get_stuff_to_save()
		success = get_reward_from_state(self.initial_state, obs, obj_stuff, self.goal_str, self.instance.env_params)
		if self.check_completed_goals:
			new_descriptions = self.find_completed_goals(obs)
			info["new_descriptions"] = new_descriptions

		self.t += 1
		return inputs, SUCCESS_REWARD if success else 0, success or self.t==self.max_episode_length, info
