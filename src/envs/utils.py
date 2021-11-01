from object2urdf import ObjectUrdfBuilder
import numpy as np
from matplotlib import pyplot as plt
import pybullet as p
import os
from src.envs.envList import *
from src.envs.reward_function import *
from src.envs.env_params import get_env_params
from src.envs.descriptions import generate_all_descriptions



def generate_urdfs(object_folder="./src/envs/ShapeNet/VEHICLE/"):
    # Build entire libraries of URDFs
    builder = ObjectUrdfBuilder(object_folder)
    builder.build_library(force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'top')


    


# # Generate urdf files for ShapeNet objects
# import os
# path = os.path.dirname(__file__)
# generate_urdfs(path + "/ShapeNet/VEHICLE/" )


# # Reconstruct the initial and final image of one episode. Finally we want the rgb_matrix for every state image of each episode.
# pixels = 600
# viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.15, 0.14, 0.15], distance=1.3, yaw=-30, pitch=-30, roll=0, upAxisIndex=2)
# projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)

# with np.load('./src/envs/collected_data/UR5/Tianwei/obs_act_etc/84/data.npz', allow_pickle=True) as data:
#     obj_stuff_data = data['obj_stuff']
#     obs_init = data['obs'][0]
#     obs_final = data['obs'][-1]
#     env_stuff_data_init = []
#     for element in obj_stuff_data:
#         env_stuff_data_init.append(element)
#     env_stuff_data_init.append(obs_init)
#     env_stuff_data_final = []
#     for element in obj_stuff_data:
#         env_stuff_data_final.append(element)
#     env_stuff_data_final.append(obs_final)
#     joint_poses_init = data['joint_poses'][0]
#     joint_poses_final = data['joint_poses'][-1]

# print(env_stuff_data_init)
# print(env_stuff_data_final)
# # print(joint_poses_init)
# # print(joint_poses_final)

# env = UR5PlayAbsRPY1Obj()

# # save all descriptions from initial state to final state
# params = get_env_params()
# train_des, test_des = sample_descriptions_from_state(obs_init, obs_final, obj_stuff_data, params)
# print("train descriptions: ", train_des)
# print("test descriptions: ", test_des)


# # save initial image of an episode
# env.reset(o=env_stuff_data_init[2], description=None, info_reset=env_stuff_data_init[:2], joint_poses = joint_poses_init) 
# img_arr_init = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
#                            renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
# plot1 = plt.figure(1)
# plt.imshow(img_arr_init)
# # save final image of an episode
# env.reset(o=env_stuff_data_final[2], description=None, info_reset=env_stuff_data_final[:2], joint_poses = joint_poses_final) 
# img_arr_final = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
#                            renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
# plot2 = plt.figure(2)
# plt.imshow(img_arr_final)
# plt.show()




# # See the goal of each episode, and check if all rewards are True
# goals = {}
# r = {}
# nb_true_reward = 0
# for i in range(101):
#     with np.load('./src/envs/collected_data/UR5/Tianwei6/obs_act_etc/' + str(i) + '/data.npz', allow_pickle=True) as data:
#         g = data['goal_str']
#         goals[i] = g
#         obj_stuff_data = data['obj_stuff']
#         obs_init = data['obs'][0]
#         obs_final = data['obs'][-1]
#         params = get_env_params()
#         reward = False
#         for gl in g:
#             reward = get_reward_from_state(obs_init, obs_final, obj_stuff_data, gl, params)
#             r[i] = reward
#             if reward == True:
#                 nb_true_reward = nb_true_reward + 1
# print(goals)
# print(r)
# print("Nb true reward: ", nb_true_reward)




# # Reconstruct the initial and final image for each episode. We would like to save the initial and final image in JPG for each recorded episode (train and test separately).
# pixels = 600
# viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.15, 0.14, 0.15], distance=1.3, yaw=-30, pitch=-30, roll=0, upAxisIndex=2)
# projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)

# for i in range(101):
#     if os.path.exists('./src/envs/collected_data/UR5/Tianwei6/obs_act_etc/' + str(i) + '/'):
#         with np.load('./src/envs/collected_data/UR5/Tianwei6/obs_act_etc/' + str(i) + '/data.npz', allow_pickle=True) as data:
#             goal = data['goal_str']
#             obj_stuff_data = data['obj_stuff']
#             obs_init = data['obs'][0]
#             obs_final = data['obs'][-1]
#             env_stuff_data_init = []
#             for element in obj_stuff_data:
#                 env_stuff_data_init.append(element)
#             env_stuff_data_init.append(obs_init)
#             env_stuff_data_final = []
#             for element in obj_stuff_data:
#                 env_stuff_data_final.append(element)
#             env_stuff_data_final.append(obs_final)
#             joint_poses_init = data['joint_poses'][0]
#             joint_poses_final = data['joint_poses'][-1]

#         env = UR5PlayAbsRPY1Obj()

#         params = get_env_params()
#         train_descriptions, test_descriptions, all_descriptions = generate_all_descriptions(params)
#         train_des, test_des = sample_descriptions_from_state(obs_init, obs_final, obj_stuff_data, params)


#         if not os.path.exists('./src/envs/dataset_images/'):
#             os.makedirs('./src/envs/dataset_images/')
#         if not os.path.exists('./src/envs/dataset_images/train/'):
#             os.makedirs('./src/envs/dataset_images/train/')
#         if not os.path.exists('./src/envs/dataset_images/test/'):
#             os.makedirs('./src/envs/dataset_images/test/')
        
        
#         if train_des:
#             count_train = len(list(os.listdir('./src/envs/dataset_images/train/')))
#             os.makedirs('./src/envs/dataset_images/train/' + str(count_train) + '/')

#             # save initial image of an episode
#             env.reset(o=env_stuff_data_init[2], description=None, info_reset=env_stuff_data_init[:2], joint_poses = joint_poses_init) 
#             img_arr_init = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
#                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
#             plt.imsave('./src/envs/dataset_images/train/' + str(count_train) + '/initial.png', img_arr_init)
#             # save final image of an episode
#             env.reset(o=env_stuff_data_final[2], description=None, info_reset=env_stuff_data_final[:2], joint_poses = joint_poses_final) 
#             img_arr_final = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
#                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
#             plt.imsave('./src/envs/dataset_images/train/' + str(count_train) + '/final.png', img_arr_final)
#         if test_des:
#             count_test = len(list(os.listdir('./src/envs/dataset_images/test/')))
#             os.makedirs('./src/envs/dataset_images/test/' + str(count_test) + '/')

#             # save initial image of an episode
#             env.reset(o=env_stuff_data_init[2], description=None, info_reset=env_stuff_data_init[:2], joint_poses = joint_poses_init) 
#             img_arr_init = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
#                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
#             plt.imsave('./src/envs/dataset_images/test/' + str(count_test) + '/initial.png', img_arr_init)
#             # save final image of an episode
#             env.reset(o=env_stuff_data_final[2], description=None, info_reset=env_stuff_data_final[:2], joint_poses = joint_poses_final) 
#             img_arr_final = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
#                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
#             plt.imsave('./src/envs/dataset_images/test/' + str(count_test) + '/final.png', img_arr_final)




# Create the train and the test dataset.
pixels = 400
viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.15, 0.14, 0.15], distance=1.3, yaw=-30, pitch=-30, roll=0, upAxisIndex=2)
projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)

for i in range(101):
    if os.path.exists('./src/envs/collected_data/UR5/Tianwei6/obs_act_etc/' + str(i) + '/'):
        with np.load('./src/envs/collected_data/UR5/Tianwei6/obs_act_etc/' + str(i) + '/data.npz', allow_pickle=True) as data:
            goal = data['goal_str']
            obj_stuff_data = data['obj_stuff']
            obs_init = data['obs'][0]
            obs_final = data['obs'][-1]
            env_stuff_data_init = []
            for element in obj_stuff_data:
                env_stuff_data_init.append(element)
            env_stuff_data_init.append(obs_init)
            env_stuff_data_final = []
            for element in obj_stuff_data:
                env_stuff_data_final.append(element)
            env_stuff_data_final.append(obs_final)
            joint_poses_init = data['joint_poses'][0]
            joint_poses_final = data['joint_poses'][-1]

        env = UR5PlayAbsRPY1Obj()

        params = get_env_params()
        train_descriptions, test_descriptions, all_descriptions = generate_all_descriptions(params)
        train_des, test_des = sample_descriptions_from_state(obs_init, obs_final, obj_stuff_data, params)


        if not os.path.exists('./src/envs/dataset/'):
            os.makedirs('./src/envs/dataset/')
        if not os.path.exists('./src/envs/dataset/train/'):
            os.makedirs('./src/envs/dataset/train/')
        if not os.path.exists('./src/envs/dataset/test/'):
            os.makedirs('./src/envs/dataset/test/')
        

        if train_des:
            count_train = len(list(os.listdir('./src/envs/dataset/train/')))
            os.makedirs('./src/envs/dataset/train/' + str(count_train) + '/')

            # save initial image of an episode
            env.reset(o=env_stuff_data_init[2], description=None, info_reset=env_stuff_data_init[:2], joint_poses = joint_poses_init) 
            img_arr_init = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
            # save final image of an episode
            env.reset(o=env_stuff_data_final[2], description=None, info_reset=env_stuff_data_final[:2], joint_poses = joint_poses_final) 
            img_arr_final = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb

            np.savez('./src/envs/dataset/train/' + str(count_train) + '/data', obs_init=obs_init, obs_final=obs_final, obj_stuff_data=obj_stuff_data, joint_poses_init=joint_poses_init, joint_poses_final=joint_poses_final, img_arr_init=img_arr_init, img_arr_final=img_arr_final, goal=goal, descriptions=train_des)
        if test_des:
            count_test = len(list(os.listdir('./src/envs/dataset/test/')))
            os.makedirs('./src/envs/dataset/test/' + str(count_test) + '/')

            # save initial image of an episode
            env.reset(o=env_stuff_data_init[2], description=None, info_reset=env_stuff_data_init[:2], joint_poses = joint_poses_init) 
            img_arr_init = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb
            # save final image of an episode
            env.reset(o=env_stuff_data_final[2], description=None, info_reset=env_stuff_data_final[:2], joint_poses = joint_poses_final) 
            img_arr_final = p.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, 
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:, :, :3]  # just the rgb

            np.savez('./src/envs/dataset/test/' + str(count_test) + '/data', obs_init=obs_init, obs_final=obs_final, obj_stuff_data=obj_stuff_data, joint_poses_init=joint_poses_init, joint_poses_final=joint_poses_final, img_arr_init=img_arr_init, img_arr_final=img_arr_final, goal=goal, descriptions=test_des)
            
