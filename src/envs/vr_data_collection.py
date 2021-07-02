# import os
# os.system(r'"D:\Sorbonne University\Stage\INRIA-FLOWERS\INRIA Internship\Env\bullet3-3.17\bullet3-3.17\bin\App_PhysicsServer_SharedMemory_VR_vs2010_x64_release.exe"')
'''
For each HTC Vive controller: 
Trigger: e[BUTTONS][33]
Big central button: e[BUTTONS][32]
Side grip: e[BUTTONS][2]
'''

'''
To teleoperate, you'll need to set up pyBullet VR https://docs.google.com/document/d/1I4m0Letbkw4je5uIBxuCfhBcllnwKojJAyYSTjHbrH8/edit?usp=sharing, 
then run the 'App_PhysicsServer_SharedMemory_VR' executable you create in that process, then run this file, which should take over SteamVR window. 
The arm will track your controller, the main trigger will close the gripper and the secondary trigger will save the trajectory you have collected. 
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

p.connect(p.SHARED_MEMORY)
# p.connect(p.GUI)
# p.connect(p.TCP, "localhost", 6667)
# p.connect(p.UDP,"192.168.86.100")
# p.connect(p.UDP, "192.168.86.10",1234)

np.set_printoptions(suppress=True)

arm = 'UR5'
if arm == 'UR5':
    print('UR5!')
    env= UR5PlayAbsRPY1Obj()
else:
    env= PandaPlayAbsRPY1Obj()

env.vr_activation()
# env.reset()
p=env.p

# env.reset(p)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP , 1)
p.setVRCameraState([0.0, -0.3, -1.1], p.getQuaternionFromEuler([0, 0, 0]))

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
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
        #GRIPPER =  e[ANALOG]
        #print(GRIPPER)
        CANCEL_BUTTON = e[BUTTONS][32]      # Big central button of the same gripper controller to cancel the currently saving trajectory
        e1 = events[0]                      # Another controller. The trigger of this controller can move everything freely (without the help of the robot arm). Therefore, we shouldn't press this trigger normally.
        SAVE_BUTTON = e1[BUTTONS][32]       # Big central button of the other controller to start/finish saving a trajectory  
        #print(e1[BUTTONS])
        return 1
        #print(p.getEulerFromQuaternion(ORI))
    except:
        return 0



CONTROLLER = 0
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6

if arm == 'UR5':
    base_path = './src/envs/collected_data/UR5/'
else:
    base_path = './src/envs/collected_data/Panda/'


demonstrator = input("Please enter the name of demonstrator:\n")
print(f'Welcome, {demonstrator} !')
demonstrator_path = base_path + demonstrator + '/'
if not os.path.exists(demonstrator_path):
    os.makedirs(demonstrator_path)

obs_act_path = demonstrator_path + 'obs_act_etc/'
env_state_path = 'C:/Users/Guillermo Valle/code/captionRL/src/envs/collected_data/UR5/' + demonstrator + '/' + 'states_and_ims/'                   # use absolute path from the data-collection PC

try:
    os.makedirs(obs_act_path)
except:
    pass

try:
    os.makedirs(env_state_path)
except:
    pass



def do_command(t,t0):
    #print(t-t0)
    #print(GRIPPER)
    #print(p.getEulerFromQuaternion(ORI))
    env.instance.runSimulation()
    state = env.instance.calc_state() 
    img_arr = state['img']
    # obs, r, done, info = env.step(np.array(action))
    targetPoses = env.instance.goto(POS, ORI, GRIPPER)
    return targetPoses, img_arr


def save_stuff(env,acts, obs, joints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception):
    # what do we care about, POS, ORI and GRIPPER?
    state = env.instance.calc_state() 
    #print(p.getEulerFromQuaternion(state['observation'][3:7]))
    
    #pos_to_save = list(np.array(POS) - state['observation'][0:3]) # actually, keep it absolute
    action = np.array(POS+ORI+[GRIPPER]) 
    ori_rpy = p.getEulerFromQuaternion(ORI)
    rel_xyz = np.array(POS)-np.array(state['observation'][0:3])
    rel_rpy = np.array(ori_rpy) - np.array(p.getEulerFromQuaternion(state['observation'][3:7]))
    action_rpy =  np.array(POS+list(ori_rpy)+[GRIPPER])
    action_rpy_rel = np.array(list(rel_xyz)+list(rel_rpy)+[GRIPPER])

    
    acts.append(action), obs.append(state['observation']), joints.append(state['joints']), acts_rpy.append(action_rpy),
    acts_rpy_rel.append(action_rpy_rel), velocities.append(state['velocity']), gripper_proprioception.append(state['gripper_proprioception'])

    # Saving images to expensive here, regen state! and saveimages there
while not get_new_command():
    pass

def save_state(env, example_path, counter):
    env.p.saveBullet(example_path + '/env_states/' + str(counter) + ".bullet")


def save(npz_path, acts, obs, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception, goal_str, images):
    print(npz_path)
    if not debugging:
        
        np.savez(npz_path + '/data', acts=acts, obs=obs, joint_poses=joints, target_poses=targetJoints, acts_rpy=acts_rpy, 
        acts_rpy_rel=acts_rpy_rel, velocities=velocities, gripper_proprioception=gripper_proprioception, goal_str=goal_str, images=images)
    print('Finish saving!')

# env.p.saveBullet(os.path.dirname(os.path.abspath(__file__)) + '/init_state.bullet') 

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
                env.reset(description=goal_str)
                p=env.p
                # time.sleep(1)
                demo_count = len(list(os.listdir(obs_act_path)))
                example_path = env_state_path + str(demo_count)
                npz_path = obs_act_path + str(demo_count)
                if not debugging:
                    os.makedirs(example_path + '/env_states')
                    os.makedirs(example_path + '/env_images')
                    os.makedirs(npz_path)
                counter = 0
                control_frequency = 25 # Hz
                t0 = time.time()
                next_time = t0 + 1/control_frequency
                # reset from init which we created (allows you to press a button on the controller and reset the env)
                # env.p.restoreState(fileName = os.path.dirname(os.path.abspath(__file__)) + '/init_state.bullet')

                acts, obs, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception, goal_trj, images = [], [], [], [], [], [], [], [], [], []
                p.addUserDebugText(text='Start saving: Goal '+str(demo_count), textPosition=[0, 1, 0.7], textColorRGB=[0, 1, 0], textSize=1)
                print('Start saving!')
            
            elif SAVE_BUTTON == 4 and SAVING == True:
                SAVE_BUTTON = 0 
                SAVING = False
                with open("./src/envs/collected_data/UR5/Recorded_Goals.txt", "a+") as f:
                    f.write(goal_str + "\n")
                    f.close()
                goal_trj.append(goal_str)
                save(npz_path, acts, obs, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception, goal_trj, images)
                p.removeUserDebugItem(itemUniqueId=1)
                p.addUserDebugText(text='Finish saving: Goal '+str(demo_count), textPosition=[0, 1, 0.7], textColorRGB=[0, 0, 1], textSize=1)
                break

            elif CANCEL_BUTTON == 4 and SAVING == True:
                CANCEL_BUTTON = 0
                SAVING = False
                shutil.rmtree(example_path)
                shutil.rmtree(npz_path)
                p.removeUserDebugItem(itemUniqueId=1)
                p.addUserDebugText(text='Cancel saving: Goal '+str(demo_count), textPosition=[0, 1, 0.7], textColorRGB=[1, 0, 0], textSize=1)
                print('Cancel saving!')
                break    

            elif SAVE_BUTTON != 4 and SAVING == True:
                t = time.time()
                if t >= next_time:
                    get_new_command()
                    if counter % 30 == 0:
                        # print(1/((1/control_frequency) + (t - next_time))) # prints the current fps
                        if not debugging:
                            thread = threading.Thread(target = save_state, name = str(counter), args = (env, example_path, counter))
                            thread.start()
                            # save_state(env,example_path,counter)
                        
                    save_stuff(env,acts, obs, joints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception)
                    target, img_arr = do_command(t,t0)
                    targetJoints.append(target)
                    # images.append(img_arr)
                    # img = Image.fromarray(img_arr, 'RGBA')            
                    # img.save(example_path + '/env_images/' + str(counter) + '.png')
                    
                    next_time = next_time + 1/control_frequency
                    counter += 1

            else:
                get_new_command()
    


            

    
    except Exception as e:
        print(e)
        if not debugging:
            shutil.rmtree(example_path)
            shutil.rmtree(npz_path)
            print('Ending Data Collection')
            break
        
    
