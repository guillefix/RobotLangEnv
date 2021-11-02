import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import gym, gym.utils, gym.utils.seeding
import pybullet as p
import pybullet_data
urdfRoot = pybullet_data.getDataPath()
from src.envs.scenes import *
from src.envs.inverseKinematics import InverseKinematicsSolver
from copy import deepcopy
import numpy as np
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


class instance():
    def __init__(self, env_params, bullet_client, offset, load_scene, arm_lower_lim, arm_upper_lim,
                 env_lower_bound, env_upper_bound,
                 obj_lower_bound, obj_upper_bound, use_orientation, return_velocity, render_scene,
                 fixed_gripper=False, num_objects=0, arm_type='UR5', description=None):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.door, self.drawer, self.pad, self.objects, self.objects_ids, self.buttons, self.toggles = load_scene(self.bullet_client, env_params,  offset, flags,
                                                                                                                  env_lower_bound, env_upper_bound, num_objects, description)

        self.num_objects = num_objects
        self.env_params = env_params
        self.use_orientation = use_orientation
        self.return_velocity = return_velocity
        self.num_objects = len(self.objects_ids)

        self.arm_lower_lim = arm_lower_lim
        self.arm_upper_lim = arm_upper_lim
        self.env_upper_bound = env_upper_bound
        self.env_lower_bound = env_lower_bound
        self.obj_lower_bound = obj_lower_bound
        self.obj_upper_bound = obj_upper_bound
        self.render_scene = render_scene
        self.physics_client_active = 0
        self.fixed_gripper = fixed_gripper
        self.arm_type = arm_type
        if self.arm_type == 'UR5':
            self.default_arm_orn_RPY = [0, 0, 0]
            self.default_arm_orn = self.bullet_client.getQuaternionFromEuler(self.default_arm_orn_RPY)
            self.init_arm_base_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
            self.endEffectorIndex = 7
            self.restJointPositions = [-1.50189075, - 1.6291067, - 1.87020409, - 1.21324173, 1.57003561, 0.06970189]
            self.numDofs = 6
            self.init_arm_base_pos = np.array([0, -0.5, 0.0])
        else:
            raise NotImplementedError
        self.ll = [-7] * self.numDofs
        self.ul = [7] * self.numDofs
        self.jr = [6] * self.numDofs
        self.record_images = False
        self.last_obs = None 
        self.last_ag = None

        sphereRadius = 0.03
        mass = 1
        colSphereId = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius)
        visId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius,
                                                     rgbaColor=[1, 0, 0, 1])
        centering_offset = np.array([0, 0.0, 0.0])
        self.original_offset = offset 

        print(currentdir)
        global ll
        global ul
        if self.arm_type == 'UR5':

            self.arm = self.bullet_client.loadURDF(currentdir + "/ur_e_description/ur5e2.urdf",
                                                   self.init_arm_base_pos + offset,
                                                   self.init_arm_base_orn, useFixedBase=True, flags=flags)
            self.IKSolver = InverseKinematicsSolver(self.init_arm_base_pos, self.init_arm_base_orn, self.endEffectorIndex, self.restJointPositions)
        else:
            raise NotImplementedError

        self.offset = offset + centering_offset  # to center the env about the gripper location

        for j in range(self.bullet_client.getNumJoints(self.arm)):
            self.bullet_client.changeDynamics(self.arm, j, linearDamping=0, angularDamping=0)

        self.reset_scene()

    def reset_scene(self):
        self.state = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.gripper_height = 0.2

        self.previous_button_state = self.extract_state(self.toggles.copy())
        self.button_state = self.previous_button_state.copy()
        self.state_buttons = dict(zip(self.previous_button_state.keys(), [False] * len(self.previous_button_state.keys())))
        self.pad_color = [float(self.state_buttons[k]) for k in [8, 10, 12]] + [1]
        pad_color = list(np.array([float(self.state_buttons[k]) for k in [8, 10, 12]]) * 0.5 + 0.5 * np.array([0.5] * 3)) + [1]
        self.bullet_client.changeVisualShape(self.pad, -1, rgbaColor=pad_color)
        
        self.state_dict = dict()
        self.state = None

    # Adds the offset (i.e to actions or objects being created) so that all instances share an action/obs space
    def add_centering_offset(self, numbers):
        # numbers either come in as xyz, or xyzxyzxyz etc (in the case of goals or achieved goals)
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers + offset
        return numbers

    # Subtract the offset (i.e when reading out states) so the obs observed have correct info
    def subtract_centering_offset(self, numbers):
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers - offset
        return numbers

    def check_on_pad(self, pos):
        return (-0.17 < pos[0] < 0.17) and (0.15-0.17 < pos[1]<0.15+0.17) and (-0.03 < pos[2] < 0.01)

    # Update color of object if set on the pad
    def update_obj_colors(self):
        for i, o in enumerate(self.objects):
            if self.previous_state_dict is not None:
                previous_pos = self.previous_state_dict['obj_{}_{}'.format(i, 'pos')]
                pos = o.position
                if self.check_on_pad(pos) and (not self.check_on_pad(previous_pos) or self.switch):
                    rgb = self.pad_color.copy()
                    for j in range(len(rgb)-1):
                        if rgb[j] == 1:
                            rgb[j] -= np.random.uniform(0, 0.2)
                        else:
                            rgb[j] += np.random.uniform(0, 0.2)
                    o.update_color(rgb_dict[str([int(c) for c in self.pad_color[:-1]])], rgb)
            else:
                pos = o.position
                if self.check_on_pad(pos):
                    rgb = self.pad_color.copy()
                    for j in range(len(rgb)-1):
                        if rgb[j] == 1:
                            rgb[j] -= np.random.uniform(0, 0.2)
                        else:
                            rgb[j] += np.random.uniform(0, 0.2)
                    o.update_color(rgb_dict[str([int(c) for c in self.pad_color[:-1]])], rgb)

    # Checks if the button or dial was pressed, and changes the environment to reflect it
    def updateToggles(self):
        self.switch = False
        for k, v in self.toggles.items():
            jointstate = self.bullet_client.getJointState(k, 0)[0]
            self.button_state[k] = jointstate
            if self.previous_button_state[k] > 0 and jointstate < 0:
                # switch color!
                self.switch = True
                self.state_buttons[k] = not self.state_buttons[k]
            if v[0] == 'button_red':
                if self.state_buttons[k]:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 0, 0, 1])
                else:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
            if v[0] == 'button_green':
                if self.state_buttons[k]:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[0, 1, 0, 1])
                else:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
            if v[0] == 'button_blue':
                # print(jointstate)
                # print(self.bullet_client.getJointState(k, 0)[1])
                if self.state_buttons[k]:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[0, 0, 1, 1])
                else:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
            self.previous_button_state[k] = jointstate
        if self.switch:
            # update pad color
            self.pad_color = [float(self.state_buttons[k]) for k in [8, 10, 12]] + [1]
            pad_color = list(np.array([float(self.state_buttons[k]) for k in [8, 10, 12]]) * 0.5  + 0.5 * np.array([0.5]*3)) + [1]
            self.bullet_client.changeVisualShape(self.pad, -1, rgbaColor=pad_color)



    def extract_state(self, toggles):
        to_save = dict()
        for k, v in toggles.items():
            jointstate = self.bullet_client.getJointState(k, 0)[0]
            to_save[k] = jointstate
        return to_save
        # Classic dict-env .step function

    # Takes environment steps s.t the sim runs at 25 Hz
    def runSimulation(self):

        self.updateToggles()  

        for i in range(0, 12):  # 25Hz control at 300
            self.bullet_client.stepSimulation()
        for o in self.objects:
            o.update_position()


    def reset_objects(self, obs=None):
        if obs is None:
            # reset 3 buttons' color
            for k, v in self.toggles.items():
                if v[0] == 'button_red':
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
                if v[0] == 'button_green':
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
                if v[0] == 'button_blue':
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])

            # reset pad color
            self.previous_button_state = self.extract_state(self.toggles.copy())
            self.button_state = self.previous_button_state.copy()
            self.state_buttons = dict(zip(self.previous_button_state.keys(), [False] * len(self.previous_button_state.keys())))
            self.pad_color = [float(self.state_buttons[k]) for k in [8, 10, 12]] + [1]
            pad_color = list(np.array([float(self.state_buttons[k]) for k in [8, 10, 12]]) * 0.5 + 0.5 * np.array([0.5] * 3)) + [1]
            self.bullet_client.changeVisualShape(self.pad, -1, rgbaColor=pad_color)
            self.bullet_client.resetBasePositionAndOrientation(self.drawer['drawer'],
                                                               self.drawer['defaults']['pos'],
                                                               self.drawer['defaults']['ori'])

            self.bullet_client.resetJointState(self.door, 0, 0.)  # reset door
            for i in self.buttons:
                self.bullet_client.resetJointState(i, 0, 0.)  # reset button etc

            for _ in range(100):
                self.bullet_client.stepSimulation()
            for o in self.objects:
                o.update_position()
        else:
            # reset pad color
            pad_color = obs[115:119] * 0.5  + 0.5 * np.array([0.5]*4)
            pad_color[3] = 1

            self.bullet_client.changeVisualShape(self.pad, -1, rgbaColor=pad_color)

            drawer_pos = self.drawer['defaults']['pos']
            drawer_pos[1] = obs[114]
            self.bullet_client.resetBasePositionAndOrientation(self.drawer['drawer'],
                                                               drawer_pos,
                                                               self.drawer['defaults']['ori'])
            self.bullet_client.resetJointState(self.door, 0, obs[113])  # reset door
            for j, i_button in enumerate(self.buttons):
                self.bullet_client.resetJointState(i_button, 0, obs[119 + j * 2])  # reset button etc
                self.state_buttons[i_button] = obs[120 + j * 2]
            self.previous_state_buttons = self.state_buttons

            for k, v in self.toggles.items():
                if v[0] == 'button_red':
                    if self.state_buttons[k]:
                        self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 0, 0, 1])
                    else:
                        self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
                if v[0] == 'button_green':
                    if self.state_buttons[k]:
                        self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[0, 1, 0, 1])
                    else:
                        self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
                if v[0] == 'button_blue':
                    if self.state_buttons[k]:
                        self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[0, 0, 1, 1])
                    else:
                        self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])

            for i, o in enumerate(self.objects):
                pos =  obs[8 + i * 35: 11 + i * 35]
                orn =  obs[11 + i * 35: 14 + i * 35]
                color = obs[37 + i * 35: 40 + i * 35]
                self.bullet_client.resetBasePositionAndOrientation(self.objects_ids[i], pos, self.bullet_client.getQuaternionFromEuler(orn))
                o.update_position()
                o.update_color(new_rgb = np.array(color.tolist() + [1]))
            stop = 1



    # Resets the arm joints to a specific pose
    def reset_arm_joints(self, arm, poses):
        index = 0

        for j in range(len(poses)):
            self.bullet_client.changeDynamics(arm, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(arm, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(arm, j, poses[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(arm, j, poses[index])
                index = index + 1

    # Resets the arm - if o is specified it will reset to that pose
    def reset_arm(self, which_arm=None, o=None, from_init=True, joint_poses=None):
        orn = self.default_arm_orn
        if o is None:
            new_pos = self.add_centering_offset(np.array([0, 0.15, 0.055]))
            if self.arm_type == 'UR5':
                new_pos[2] = new_pos[2] + 0.2
        else:
            new_pos = self.add_centering_offset(o[0:3])
            if self.use_orientation:
                if self.return_velocity:
                    orn = o[6:10]  # because both pos and pos_vel are in the state
                else:
                    orn = o[3:7]

        if from_init:
            self.reset_arm_joints(which_arm, self.restJointPositions)  # put it into a good init for IK

        jointPoses = self.bullet_client.calculateInverseKinematics(which_arm, self.endEffectorIndex, new_pos, orn, )[0:6]
        self.reset_arm_joints(which_arm, jointPoses)
        # Reset arm joints
        if joint_poses is not None:
            for j in range(8):
                self.bullet_client.resetJointState(self.arm, j, joint_poses[j])
        # Reset the left and right gripper
        if o is not None:
            self.bullet_client.resetJointState(self.arm, 18, o[7]/23)
            self.bullet_client.resetJointState(self.arm, 20, o[7]/23)

    # Overall reset function, passes o to the above specific reset functions if sepecified
    def reset(self, o=None, info_reset=None, description=None, joint_poses=None):
        self.bullet_client.removeAllUserDebugItems()      ###### Comment this when running interactive.py
        self.state_dict = dict()
        self.state = None
        self.previous_state = None
        self.previous_state_dict = None

        # resample objects
        for obj_id in self.objects_ids:
            self.bullet_client.removeBody(obj_id)

        self.objects, self.objects_ids, self.objects_added = sample_objects(description, self.bullet_client, self.env_params, self.num_objects, info_reset)
        # print("Arm nb joints: ", self.bullet_client.getNumJoints(self.arm))
        self.reset_arm(which_arm=self.arm, o=o, joint_poses=joint_poses)
        self.reset_objects(o)
        for o in self.objects:
            o.update_position()
        self.update_obj_colors()        ###### Comment this when regenerating the scene/image
        self.t = 0
        self.bullet_client.addUserDebugText(text=description, textPosition=[0, 1, 0.8], textColorRGB=[1, 1, 1], textSize=1.4)        ###### Comment this when regenerating the scene/image OR when running interactive.py

    def get_stuff_to_save(self):
        to_save = [self.objects_added, [o.size_encoding for o in self.objects]]
        return to_save
    
    # Binary return indicating if something is between the gripper prongs
    def gripper_proprioception(self):
        if self.arm_type == 'UR5':
            gripper_one = np.array(self.bullet_client.getLinkState(self.arm, 18)[0])
            gripper_two = np.array(self.bullet_client.getLinkState(self.arm, 20)[0])
            ee = np.array(self.bullet_client.getLinkState(self.arm, self.endEffectorIndex)[0])
            wrist = np.array(self.bullet_client.getLinkState(self.arm, self.endEffectorIndex - 1)[0])
            avg_gripper = (gripper_one + gripper_two) / 2
            point_one = ee - (ee - wrist) * 0.5  # far up
            point_two = avg_gripper + (ee - wrist) * 0.2  # between the prongs

            try:
                obj_id, link_index, hit_fraction, hit_position, hit_normal = self.bullet_client.rayTest(point_one, point_two)[0]

                # visualises the ray getting tripped
                # self.bullet_client.addUserDebugLine(gripper_one, gripper_two, [1,0,0], 0.5, 1)

                if hit_fraction == 1.0 or link_index == 18 or link_index == 20:
                    return 0  # nothing in the hand
                else:
                    return 1  # something in the hand
            except:
                return -1  # this shouldn't ever happen because the ray will always hit the other side of the gripper
        else:
            return -1

    # Calculates the state of the arm
    def calc_actor_state(self):
        '''
        Returns a dict of all the pieces of information you could want (e.g pos, orn, vel, orn vel, gripper, joints)
        '''
        state = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)
        pos, orn, vel, orn_vel = state[0], state[1], state[-2], state[-1]

        gripper_state = [self.bullet_client.getJointState(self.arm, 18)[0] * 23]  # put it on a 0-1 scale

        joint_poses = [self.bullet_client.getJointState(self.arm, j)[0] for j in range(8)]

        return {'pos': self.subtract_centering_offset(pos), 'orn': orn, 'pos_vel': vel, 'orn_vel': orn_vel,
                'gripper': gripper_state, 'joints': joint_poses, 'proprioception': self.gripper_proprioception()}



    # Combines actor and environment state into vectors, and takes all the different slices one could want in a return dict
    def calc_state(self):
        # arm_state: 0 - 8
        # obj_0_pos: 8 - 11
        # obj_0_orn: 11 - 14
        # obj_0_type: 14 - 37
        # obj_0_color: 37 - 40
        # obj_0_size: 40 - 43
        # obj_1_pos: 43 - 46
        # obj_1_orn: 46 - 49
        # obj_1_type: 49 - 72
        # obj_1_color: 72 - 75
        # obj_1_size: 75 - 78
        # obj_2_pos: 78 - 81
        # obj_2_orn: 81 - 84
        # obj_2_type: 84 - 107
        # obj_2_color: 107 - 110
        # obj_2_size: 110 - 113
        # door_pos: 113 - 114
        # drawer_pos: 114 - 115
        # pad_color: 115 - 119
        # button_0: 119 - 120
        # button_0_state: 120 - 121
        # button_1: 121 - 122
        # button_1_state: 122 - 123
        # button_2: 123 - 124
        # button_2_state: 124 - 125
        if self.state is not None:
            self.previous_state_dict = deepcopy(self.state_dict)
            self.previous_state = self.state.copy()

        self.update_obj_colors() # paint object

        arm_state = self.calc_actor_state()
        arm_elements = ['pos']
        if self.return_velocity:
            arm_elements.append('pos_vel')
        if self.use_orientation:
            arm_elements.append('orn')
        arm_elements.append('gripper')
        state = np.concatenate([np.array(arm_state[i]) for i in arm_elements])
        self.state_dict['arm_state'] = state.copy()

        # get objects features
        objects_features = [o.get_features() for o in self.objects]
        object_states = dict()
        for i, o in enumerate(objects_features):
            for k, v in o.items():
                object_states['obj_{}_{}'.format(i, k)] = v
                self.state_dict['obj_{}_{}'.format(i, k)] = v

        self.state_dict['door_pos'] = np.array([self.bullet_client.getJointState(self.door, 0)[0]])
        self.state_dict['drawer_pos'] = np.array([self.bullet_client.getBasePositionAndOrientation(self.drawer['drawer'])[0][1]])  # get the y pos
        self.state_dict['pad_color'] = np.array(self.pad_color)
        for j in range(len(self.buttons)):
            self.state_dict['button_{}'.format(j)] = np.array([self.bullet_client.getJointState(self.buttons[j], 0)[0]])
            self.state_dict['button_{}_state'.format(j)] = np.array([float(self.state_buttons[self.buttons[j]])])


        if self.record_images:
            img_arr = self.bullet_client.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=self.bullet_client.ER_NO_SEGMENTATION_MASK, shadow=0,
                                                        renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
        else:
            img_arr = None

        # fix flipping of PyBullet randomly returns equivalent quaternions (e.g, one timestep it will return the -ve of the previous quat)
        # For a smooth state signal, flip the quaternion if all elements are -ive of prev step
        for k, v in self.state_dict.items():
            if 'orientation' in k:
                quat = v
                last_quat = self.previous_state_dict[k]
                if (np.sign(quat) == -np.sign(last_quat)).all():
                    self.state_dict[k] = -v

        state = np.concatenate(list(self.state_dict.values()))


        self.state_dict_euler = deepcopy(self.state_dict)
        for k, v in sorted(self.state_dict.items()):
            if 'orn' in k:
                self.state_dict_euler[k] = np.array(p.getEulerFromQuaternion(v))
        state_euler = np.concatenate(list(self.state_dict_euler.values()))

        return_dict = {
            'obs_quat': state.copy().astype('float32'),
            # just the x,y,z pos of the self, the controllable aspects
            'joints': arm_state['joints'],
            'velocity': np.concatenate([arm_state['pos_vel'], arm_state['orn_vel']]),
            'img': img_arr,
            'observation': state_euler.copy(),
            'gripper_proprioception': arm_state['proprioception']
        }
        
        self.state = return_dict.copy()
        if self.previous_state == None:
            self.previous_state = self.state
        return return_dict

    # PyBullet randomly returns equivalent quaternions (e.g, one timestep it will return the -ve of the previous quat)
    # For a smooth state signal, flip the quaternion if all elements are -ive of prev step
    def quaternion_safe_the_obs(self, obs, ag):
        def flip_quats(vector, last, idxs):
            for pair in idxs:
                quat = vector[pair[0]:pair[1]]
                last_quat = last[pair[0]:pair[1]]
                # print(np.sign(quat) == -np.sign(last_quat))
                if (np.sign(quat) == -np.sign(last_quat)).all():  # i.e, it is an equivalent quaternion
                    vector[pair[0]:pair[1]] = - vector[pair[0]:pair[1]]
            return vector

        if self.last_obs is None:
            pass
        else:
            indices = [(3, 7), (11, 15)]  # self, and object one xyz q1-4 grip xyz q1-4
            if self.num_objects == 2:
                indices.append((19, 23))
            obs = flip_quats(obs, self.last_obs, indices)
            indices = [(3, 7)]  # just the objeect
            if self.num_objects == 2:
                indices.append((10, 14))
            ag = flip_quats(ag, self.last_ag, indices)

        self.last_obs = obs
        self.last_ag = ag
        return obs, ag

    def render(self, mode='human'):

        if (mode == "human"):
            self.render_scene = True
            return np.array([])
        if mode == 'rgb_array':
            raise NotImplementedError

    def close(self):
        print('closing')
        self.bullet_client.disconnect()

    def _seed(self, seed=None):
        print('seeding')
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        return [seed]

    # Understand what type of action was commanded
    def perform_action(self, action, action_type):
        '''
        Takes in the action, and uses the appropriate function to determine to get joint angles and perform the action in the environment
        '''
        if action_type == 'absolute_quat':
            targetPoses = self.absolute_quat_step(action)
        elif action_type == 'relative_quat':
            targetPoses = self.relative_quat_step(action)
        elif action_type == 'relative_joints':
            targetPoses = self.relative_joint_step(action)
        elif action_type == 'absolute_joints':
            targetPoses = self.absolute_joint_step(action)
        elif action_type == 'absolute_rpy':
            targetPoses = self.absolute_rpy_step(action)
        elif action_type == 'relative_rpy':
            targetPoses = self.relative_rpy_step(action)
        else:
            raise NotImplementedError
        return targetPoses

    def absolute_quat_step(self, action):
        assert len(action) == 8

        new_pos = action[0:3]
        gripper = action[-1]
        targetPoses = self.goto(new_pos, action[3:7], gripper)
        return targetPoses

    def relative_quat_step(self, action):
        assert len(action) == 8

        current_pos = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[0]
        current_orn = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[1]
        new_pos = action[0:3] + current_pos
        new_orn = action[3:7] + current_orn
        gripper = action[-1]
        targetPoses = self.goto(new_pos, new_orn, gripper)
        return targetPoses

    def absolute_rpy_step(self, action):
        assert len(action) == 7
        new_pos = action[0:3]
        new_orn = action[3:6]
        gripper = action[-1]
        targetPoses = self.goto(new_pos, self.bullet_client.getQuaternionFromEuler(new_orn), gripper)
        return targetPoses

    def relative_rpy_step(self, action):
        assert len(action) == 7
        current_pos = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[0]
        current_orn = self.bullet_client.getEulerFromQuaternion(self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[1])
        new_pos = action[0:3] + current_pos
        new_orn = action[3:6] + current_orn
        gripper = action[-1]
        targetPoses = self.goto(new_pos, self.bullet_client.getQuaternionFromEuler(new_orn), gripper)
        return targetPoses

    def relative_joint_step(self, action):
        current_poses = np.array([self.bullet_client.getJointState(self.arm, j)[0] for j in range(self.numDofs)])
        jointPoses = action[:-1] + current_poses
        gripper = action[-1]
        targetPoses = self.goto_joint_poses(jointPoses, gripper)
        return targetPoses

    def absolute_joint_step(self, action):
        targetPoses = self.goto_joint_poses(action[:-1], action[-1])
        return targetPoses

    # Convert these to xyz quat if they are not joints
    def goto(self, pos=None, orn=None, gripper=None):
        '''
        Uses PyBullet IK to solve for desired joint angles
        '''
        if pos is not None and orn is not None:

            pos = self.add_centering_offset(pos)
            current_poses = np.array([self.bullet_client.getJointState(self.arm, j)[0] for j in range(self.numDofs)])
            jointPoses = self.IKSolver.calc_angles(pos, orn, current_poses)
            targetPoses = self.goto_joint_poses(jointPoses, gripper)
        return targetPoses

    # Send joint commands to the robot - by default UR5 uses a support function in 'inverseKinematics.py' for greater stability
    def goto_joint_poses(self, jointPoses, gripper):
        indexes = [i for i in range(self.numDofs)]
        index_len = len(indexes)
        local_ll, local_ul = None, None
        local_ll = np.array([-np.pi * 2] * self.numDofs)
        local_ul = np.array([-0.7, np.pi * 2, -0.5, np.pi * 2, np.pi * 2, np.pi * 2])
        inc = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
        targetPoses = np.clip(np.array(jointPoses[:index_len]), local_ll[:index_len], local_ul[:index_len])

        current_poses = np.array([self.bullet_client.getJointState(self.arm, j)[0] for j in range(index_len)])
        targetPoses = np.clip(targetPoses, current_poses - inc, current_poses + inc)
        self.bullet_client.setJointMotorControlArray(self.arm, indexes, self.bullet_client.POSITION_CONTROL,
                                                     targetPositions=targetPoses,
                                                     forces=[240.] * len(indexes))

        if gripper is not None:
            self.close_gripper(gripper)

        return targetPoses

    # Command to control the gripper from 0-1
    def close_gripper(self, amount):
        '''
        0 : open gripper
        1 : close gripper
        '''
        # left/ right driver appears to close at 0.03
        amount -= 0.2
        driver = amount * 0.055

        self.bullet_client.setJointMotorControl2(self.arm, 18, self.bullet_client.POSITION_CONTROL, driver,
                                                    force=100)
        left = self.bullet_client.getJointState(self.arm, 18)[0]
        self.bullet_client.setJointMotorControl2(self.arm, 20, self.bullet_client.POSITION_CONTROL, left,
                                                    force=1000)

        spring_link = amount * 0.5
        self.bullet_client.setJointMotorControl2(self.arm, 12, self.bullet_client.POSITION_CONTROL, spring_link,
                                                    force=100)
        self.bullet_client.setJointMotorControl2(self.arm, 15, self.bullet_client.POSITION_CONTROL, spring_link,
                                                    force=100)

        driver_mimic = amount * 0.8
        self.bullet_client.setJointMotorControl2(self.arm, 10, self.bullet_client.POSITION_CONTROL, driver_mimic,
                                                    force=100)
        self.bullet_client.setJointMotorControl2(self.arm, 13, self.bullet_client.POSITION_CONTROL, driver_mimic,
                                                    force=100)
