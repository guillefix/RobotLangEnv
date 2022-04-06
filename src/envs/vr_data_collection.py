'''
For each HTC Vive controller:
Trigger: e[BUTTONS][33]
Big central button: e[BUTTONS][32]
Side grip: e[BUTTONS][2]
'''

'''
To teleoperate, you'll need to set up pyBullet VR https://docs.google.com/document/d/1I4m0Letbkw4je5uIBxuCfhBcllnwKojJAyYSTjHbrH8/edit?usp=sharing,
then run the 'App_PhysicsServer_SharedMemory_VR' executable you create in that process, then run this file, which should take over SteamVR window.
Please read and follow the "VR Setup" and "VR Instructions for the Demonstrator".
We save 'full state' not images during data collection - because this allows us to determinstically reset the environment to that state and then
collect images from any angle desired!
'''

debugging = False

import socket
import pybullet as p
import time
import pybullet_data
import numpy as np
from pickle import dumps
import math
import os
import shutil
import threading
import matplotlib.pyplot as plt
from PIL import Image
from src.envs.envList import *
from src.envs.env_params import get_env_params
from src.envs.descriptions import generate_all_descriptions
from src.envs.utils import save_traj

p.connect(p.SHARED_MEMORY)

np.set_printoptions(suppress=True)

arm = 'UR5'
env= UR5PlayAbsRPY1Obj()

env.vr_activation()
p=env.p

p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_VR_PICKING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP , 1)
p.setVRCameraState([0.0, -0.3, -1.1], p.getQuaternionFromEuler([0, 0, 0]))

# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)

POS = [1, 1, 1]
ORI = list(p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]))
GRIPPER = 0.0
SAVE_BUTTON = None          # the button to save a trajectory
CANCEL_BUTTON = None        # the button to cancel the currently saving trajectory
SAVING = False              # the current state of saving/not saving

def get_new_command():
    try:

        global POS
        global ORI
        global GRIPPER
        global SAVE_BUTTON
        global CANCEL_BUTTON
        events = p.getVREvents()
        e = events[1]         # one controller
        POS = list(e[POSITION])
        ORI = list(e[ORIENTATION])
        if e[ANALOG] > GRIPPER:
            GRIPPER = min(GRIPPER + 0.25,e[ANALOG])
        else: # i.e, force it to slowly open/close
            GRIPPER = max(GRIPPER - 0.25,e[ANALOG])
        CANCEL_BUTTON = e[BUTTONS][32]      # Big central button of the same gripper controller to cancel the currently saving trajectory
        e1 = events[0]                      # Another controller. The trigger of this controller can move everything freely (without the help of the robot arm). Therefore, we shouldn't press this trigger normally.
        SAVE_BUTTON = e1[BUTTONS][32]       # Big central button of the other controller to start/finish saving a trajectory
        return 1

    except:
        return 0



CONTROLLER = 0
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6

base_path = './src/envs/collected_data/UR5/'

demonstrator = input("Please enter the name of demonstrator:\n")
print(f'Welcome, {demonstrator} !')
demonstrator_path = base_path + demonstrator + '/'
if not os.path.exists(demonstrator_path):
    os.makedirs(demonstrator_path)

obs_act_path = demonstrator_path + 'obs_act_etc/'

try:
    os.makedirs(obs_act_path)
except:
    pass


def do_command(t,t0):
    env.instance.runSimulation()
    targetPoses = env.instance.goto(POS, ORI, GRIPPER)
    return targetPoses


def save_stuff(env,acts, obs, joints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception):
    state = env.instance.calc_state()
    action = np.array(POS+ORI+[GRIPPER])
    ori_rpy = p.getEulerFromQuaternion(ORI)
    rel_xyz = np.array(POS)-np.array(state['observation'][0:3])
    rel_rpy = np.array(ori_rpy) - np.array(p.getEulerFromQuaternion(state['observation'][3:7]))
    action_rpy =  np.array(POS+list(ori_rpy)+[GRIPPER])
    action_rpy_rel = np.array(list(rel_xyz)+list(rel_rpy)+[GRIPPER])


    acts.append(action), obs.append(state['observation']), joints.append(state['joints']), acts_rpy.append(action_rpy),
    acts_rpy_rel.append(action_rpy_rel), velocities.append(state['velocity']), gripper_proprioception.append(state['gripper_proprioception'])

while not get_new_command():
    pass




while(1):
    try:
        while(1):
            if SAVE_BUTTON == 4 and SAVING == False:
                SAVE_BUTTON = 0
                SAVING = True
                env_params = get_env_params()
                _, _, all_descriptions = generate_all_descriptions(env_params)
                goal_str = np.random.choice(all_descriptions)
                with open("./src/envs/collected_data/UR5/Recorded_Goals.txt", "a+") as f:
                    while goal_str in f.read():
                        goal_str = np.random.choice(all_descriptions)
                    f.close()

                print(goal_str)

                ###     Regenerate a specific saved initial scene in VR     ###
                # with np.load('./src/envs/collected_data/UR5/tianwei/obs_act_etc/28/data.npz', allow_pickle=True) as data:
                #     obj_stuff_data = data['obj_stuff']
                #     obs_init = data['obs'][0]
                #     env_stuff_data = obj_stuff_data + [obs_init]
                # env.reset(o=env_stuff_data[2], description=None, info_reset=env_stuff_data[:2])

                ###     New episode data collection     ###
                env.reset(description=goal_str)

                p=env.p
                # time.sleep(1)
                demo_count = len(list(os.listdir(obs_act_path)))
                npz_path = obs_act_path + str(demo_count)
                if not debugging:
                    os.makedirs(npz_path)
                counter = 0
                control_frequency = 25 # Hz
                t0 = time.time()
                next_time = t0 + 1/control_frequency

                acts, obs, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception, goal_trj, obj_stuff = [], [], [], [], [], [], [], [], [], []

                state = env.instance.calc_state()
                obj_stuff = env.instance.get_stuff_to_save()

                p.addUserDebugText(text='Start saving: Goal '+str(demo_count), textPosition=[0, 1, 0.7], textColorRGB=[0, 1, 0], textSize=1.2)
                print('Start saving!')

            elif SAVE_BUTTON == 4 and SAVING == True:
                SAVE_BUTTON = 0
                SAVING = False
                with open("./src/envs/collected_data/UR5/Recorded_Goals.txt", "a+") as f:
                    f.write(goal_str + "\n")
                    f.close()
                goal_trj.append(goal_str)
                save_traj(npz_path, acts, obs, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception, goal_trj, obj_stuff, debugging)
                p.removeUserDebugItem(itemUniqueId=1)
                p.addUserDebugText(text='Finish saving: Goal '+str(demo_count), textPosition=[0, 1, 0.7], textColorRGB=[0, 0, 1], textSize=1.2)
                break

            elif CANCEL_BUTTON == 4 and SAVING == True:
                CANCEL_BUTTON = 0
                SAVING = False
                shutil.rmtree(npz_path)
                p.removeUserDebugItem(itemUniqueId=1)
                p.addUserDebugText(text='Cancel saving: Goal '+str(demo_count), textPosition=[0, 1, 0.7], textColorRGB=[1, 0, 0], textSize=1.2)
                print('Cancel saving!')
                break

            elif SAVE_BUTTON != 4 and SAVING == True:
                t = time.time()
                if t >= next_time:
                    get_new_command()

                    save_stuff(env,acts, obs, joints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception)
                    target = do_command(t,t0)
                    targetJoints.append(target)

                    next_time = next_time + 1/control_frequency
                    counter += 1

            else:
                get_new_command()


    except Exception as e:
        print(e)
        if not debugging:
            shutil.rmtree(npz_path)
            print('Ending Data Collection')
            break
