
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import gym, gym.utils, gym.utils.seeding
import pybullet as p
import numpy as np
import pybullet_data
import time
from pybullet_utils import bullet_client
import pybullet_data as pd
urdfRoot = pybullet_data.getDataPath()
import gym.spaces as spaces
import math
from src.envs.scenes import *
from src.envs.playRewardFunc import success_func
from src.envs.inverseKinematics import InverseKinematicsSolver
from src.envs.instance import instance

lookat = [0, 0.0, 0.0]
distance = 0.8
yaw = 130
pitch = -130
pixels = 200
# viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=lookat, distance=distance, yaw=yaw, pitch=pitch, roll=0,
#                                                  upAxisIndex=2)
viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0.25, 0], distance=1.3, yaw=-30, pitch=-30, roll=0,
                                                 upAxisIndex=2)
projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)

# A function used to get camera images from the gripper - experimental
def gripper_camera(bullet_client, pos, ori):

    # Center of mass position and orientation (of link-7)
    pos = np.array(pos)
    ori = p.getEulerFromQuaternion(ori) + np.array([0,-np.pi/2,0])
    ori = p.getQuaternionFromEuler(ori)
    rot_matrix = bullet_client.getMatrixFromQuaternion(ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    camera_target = (1, 0, 0) # z-axis
    init_up_vector = (0, 0, 1) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(camera_target)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix_gripper = bullet_client.computeViewMatrix(pos, pos + camera_vector, up_vector)
    img = bullet_client.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix,shadow=0, flags = bullet_client.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return img


'''
The main gym env.
This was originally set up to support RL, so that you could have many many instances (1 arm + tabltop env) at different positions in the 
world (using the 'offset' arg each time you created one). It currently only supports one instance, but would be easy to adapt to multiple instances
with loops inside step, activate physics client, and compute reward functions
'''
class playEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, num_objects = 0, env_range_low = [-0.18, -0.18,-0.05 ], env_range_high = [0.18, 0.18, 0.15], goal_range_low = [-0.18, -0.18, -0.05], goal_range_high = [0.18, 0.18, 0.05],
                 obj_lower_bound = [-0.18, -0.18, -0.05], obj_upper_bound = [-0.18, -0.18, -0.05], sparse=True, use_orientation=False,
                 sparse_rew_thresh=0.05, fixed_gripper = False, return_velocity=True, max_episode_steps=250, 
                 play=False, action_type = 'absolute_rpy', show_goal=True, arm_type= 'Panda'): # action type can be relative, absolute, or joint relative
        fps = 300
        self.timeStep = 1. / fps
        self.render_scene = False
        self.physics_client_active = 0
        self.num_objects  = num_objects
        self.use_orientation = use_orientation
        self.return_velocity = return_velocity
        self.fixed_gripper = fixed_gripper
        self.sparse_reward_threshold = sparse_rew_thresh
        self.num_goals = max(self.num_objects, 1)
        self.play = play
        self.action_type = action_type
        self.show_goal = show_goal
        self.arm_type = arm_type
        obs_dim = 8
        self.sparse_rew_thresh = sparse_rew_thresh
        self._max_episode_steps = max_episode_steps

        obs_dim += 7 * num_objects  # pos and vel of the other pm that we are knocking around.
        # TODO actually clip input actions by this amount!!!!!!!!
        pos_step = 0.015
        orn_step = 0.1
        if action_type == 'absolute_quat':
            pos_step = 1.0
            if self.use_orientation:
                high = np.array([pos_step,pos_step,pos_step,1,1,1,1,1]) # use absolute orientations
            else:
                high = np.array([pos_step, pos_step, pos_step, 1])
        elif action_type == 'relative_quat':
            high = np.array([1, 1, 1, 1, 1, 1,1, 1])
        elif action_type == 'relative_joints':
            if self.arm_type == 'UR5':
                high = np.array([1,1,1,1,1,1, 1])
            else:
                high = np.array([1,1,1,1,1,1,1, 1])
        elif action_type == 'absolute_joints':
            if self.arm_type == 'UR5':
                high = np.array([6, 6, 6, 6, 6, 6, 1])
            else:
                high = np.array([6, 6, 6, 6, 6, 6, 6, 1])
        elif action_type == 'absolute_rpy':
            high = np.array([6, 6, 6, 6, 6, 6, 1])
        elif action_type == 'relative_rpy':
            high = np.array([1,1,1,1,1,1, 1])
        else:
            if self.use_orientation:
                high = np.array([pos_step, pos_step, pos_step, orn_step,orn_step,orn_step, 1])
            else:
                high = np.array([pos_step, pos_step, pos_step, 1])
        self.action_space = spaces.Box(-high, high)
        

        self.env_upper_bound = np.array(env_range_high)
        self.env_lower_bound = np.array(env_range_low)
        #self.env_lower_bound[1] = 0  # set the y (updown) min to 0.
        self.goal_upper_bound = np.array(goal_range_high)
        self.goal_lower_bound = np.array(goal_range_low)
        #self.goal_lower_bound[1] = 0  # set the y (updown) min to 0.

        self.obj_lower_bound = obj_lower_bound
        self.obj_upper_bound = obj_upper_bound

        if use_orientation:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1, 0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1, 0.0])])
            arm_upper_obs_lim = np.concatenate(
                [self.env_upper_bound, np.array([1, 1, 1, 1, 1, 1, 1, 0.04])])  # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 1, 1, 1, 1, 0.0])])
            obj_upper_lim = np.concatenate([self.obj_upper_bound, np.ones(7)]) # velocity and orientation
            obj_lower_lim = np.concatenate([self.obj_lower_bound, -np.ones(7)]) # velocity and orientation
            obj_upper_positional_lim = np.concatenate([self.env_upper_bound, np.ones(4)])
            obj_lower_positional_lim = np.concatenate([self.env_lower_bound, -np.ones(4)])
        else:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([0.0])])
            arm_upper_obs_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 0.04])])  # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 0.0])])
            obj_upper_lim = np.concatenate([self.obj_upper_bound, np.ones(3)])
            obj_lower_lim =  np.concatenate([self.obj_lower_bound, -np.ones(3)])
            obj_upper_positional_lim = self.env_upper_bound
            obj_lower_positional_lim = self.env_lower_bound

        upper_obs_dim = np.concatenate([arm_upper_obs_lim] + [obj_upper_lim] * self.num_objects)
        lower_obs_dim = np.concatenate([arm_lower_obs_lim] + [obj_lower_lim] * self.num_objects)
        upper_goal_dim = np.concatenate([self.env_upper_bound] * self.num_goals)
        lower_goal_dim = np.concatenate([self.env_lower_bound] * self.num_goals)

        lower_full_positional_state = np.concatenate([self.arm_lower_lim] + [obj_lower_positional_lim] * self.num_objects) # like the obs dim, but without velocity.
        upper_full_positional_state = np.concatenate([self.arm_upper_lim] + [obj_upper_positional_lim] * self.num_objects)

        #self.action_space = spaces.Box(self.arm_lower_lim, self.arm_upper_lim)

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
            achieved_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
            observation=spaces.Box(lower_obs_dim, upper_obs_dim),
            controllable_achieved_goal=spaces.Box(self.arm_lower_lim, self.arm_upper_lim),
            full_positional_state=spaces.Box( lower_full_positional_state, upper_full_positional_state)
        ))


        # if sparse:
        #     self.compute_reward = self.compute_reward_sparse

    # Resets the instances until reward is not satisfied
    def reset(self, o = None, vr =None):

        if not self.physics_client_active:
            self.activate_physics_client(vr)
            self.physics_client_active = True

        self.instance.reset(o)
        obs = self.instance.calc_state()

        # r = 0
        # while r > -1:
        #     # reset again if we init into a satisfied state
        #
        #     self.instance.reset(o)
        #     obs = self.instance.calc_state()
        #     r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        return obs

    # Directly reset the goal position
    # def reset_goal_pos(self, goal):
    #     self.instance.reset_goal_pos(goal)

    # Sets whether to render the scene
    # Img will be returned as part of the state which comes from obs, 
    # rather than from calling .render(rgb) every time as in some other envs
    def render(self, mode):
        if (mode == "human"):
            self.render_scene = True
            return np.array([])
        if mode == 'rgb_array':
            self.instance.record_images = True
        if mode == 'playback':
            self.instance.record_images = True


    # Classic dict-env .step function
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        targetPoses = self.instance.perform_action(action, self.action_type)
        self.instance.runSimulation()
        obs = self.instance.calc_state()
        if self.play:
            self.instance.update_obj_colors()
        # r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        done = False
        # success = 0 if r < 0 else 1
        return obs, 0, done, {'is_success': 0, 'target_poses': targetPoses}

    # Activates the GUI or headless physics client, and creates arm instance within it
    # Within this function is the call which selects which scene 'no obj, one obj, play etc - defined in scenes.py'
    def activate_physics_client(self, vr=None):

        if self.render_scene:
            if vr is None:
                self.p = bullet_client.BulletClient(connection_mode=p.GUI)
                #self.p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
                self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            else:
                # Trying to rig up VR
                self.p =  bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)

        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, -9.8)

        if self.play:
            scene = complex_scene
        else:
            if self.num_objects == 0 :
                scene = default_scene
            elif self.num_objects == 1:
                scene = push_scene
        self.instance = instance(self.p, [0, 0, 0], scene,  self.arm_lower_lim, self.arm_upper_lim,
                                        self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                        self.goal_upper_bound, self.obj_lower_bound, self.obj_upper_bound,  self.use_orientation, self.return_velocity,
                                         self.render_scene, fixed_gripper=self.fixed_gripper, 
                                        play=self.play, show_goal = self.show_goal, num_objects=self.num_objects, arm_type=self.arm_type)
        self.instance.control_dt = self.timeStep
        self.p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    # With VR, a physics server is already created - this connects to it
    def vr_activation(self, vr=None):
        self.p = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)

        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, -9.8)
        scene = complex_scene
        self.instance = instance(self.p, [0, 0, 0], scene, self.arm_lower_lim, self.arm_upper_lim,
                                  self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                  self.goal_upper_bound, self.obj_lower_bound, self.obj_upper_bound,
                                  self.use_orientation, self.return_velocity,
                                  self.render_scene, fixed_gripper=self.fixed_gripper,
                                  play=self.play, num_objects=self.num_objects, arm_type=self.arm_type)
        self.instance.control_dt = self.timeStep
        self.physics_client_active = True

    # def calc_target_distance(self, achieved_goal, desired_goal):
    #     distance = np.linalg.norm(achieved_goal - desired_goal)
    #     return distance

    # A basic dense reward metric, distance from goal state
    # def compute_reward(self, achieved_goal, desired_goal):
    #     return -self.calc_target_distance(achieved_goal,desired_goal)

    # A piecewise reward function, for each object it determines if all dims are within the threshold and increments reward if so
    # def compute_reward_sparse(self, achieved_goal, desired_goal, info=None):
    #     if self.play:
    #         # This success function lives in 'playRewardFunc.py'
    #         return success_func(achieved_goal, desired_goal)
    #     else:
    #         initially_vectorized = True
    #         dimension = 3
    #         if len(achieved_goal.shape) == 1:
    #             achieved_goal = np.expand_dims(np.array(achieved_goal), axis=0)
    #             desired_goal = np.expand_dims(np.array(desired_goal), axis=0)
    #             initially_vectorized = False
    #
    #         reward = np.zeros(len(achieved_goal))
    #         # only compute reward on pos not orn for the moment
    #         g_ag = 0 # increments of dimension, then skip 4 for ori
    #         g_dg = 0 # increments of dimension,
    #         for g in range(0, self.num_goals):  # piecewise reward
    #             current_distance = np.linalg.norm(achieved_goal[:, g_ag:g_ag + dimension] - desired_goal[:, g_dg:g_dg + dimension],
    #                                             axis=1)
    #             reward += np.where(current_distance > self.sparse_rew_thresh, -1, -current_distance)
    #             g_ag += dimension+ 4 # for ori
    #             g_dg += dimension
    #
    #         if not initially_vectorized:
    #             return reward[0]
    #         else:
    #             return reward

    # Env level sub goal visualisation and deletion 
    # def visualise_sub_goal(self, sub_goal, sub_goal_state = 'full_positional_state'):
    #     self.instance.visualise_sub_goal(sub_goal, sub_goal_state = sub_goal_state)
    #
    # def delete_sub_goal(self):
    #     try:
    #         self.instance.delete_sub_goal()
    #     except:
    #         pass
