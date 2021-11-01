import numpy as np
from matplotlib import pyplot as plt
import pybullet as p
import os
from src.envs.envList import *
from src.envs.reward_function import *
from src.envs.env_params import get_env_params
from src.envs.descriptions import generate_all_descriptions



# Goal distribution of all train and test descriptions
params = get_env_params()
train_descriptions, test_descriptions, all_descriptions = generate_all_descriptions(params)

# All train goal distribution
throw = 0
open = 0
close = 0
grasp = 0
move = 0
put = 0
hide = 0
turn_on = 0
turn_off = 0
make = 0
paint = 0
for train in train_descriptions:
    words = train.split(' ')
    if words[0] == 'Throw':
        throw = throw + 1
    elif words[0] == 'Open':
        open = open + 1
    elif words[0] == 'Close':
        close = close + 1
    elif words[0] == 'Grasp':
        grasp = grasp + 1
    elif words[0] == 'Move':
        move = move + 1
    elif words[0] == 'Put':
        put = put + 1
    elif words[0] == 'Hide':
        hide = hide + 1
    elif words[0] == 'Turn':
        if words[1] == 'on':
            turn_on = turn_on + 1
        elif words[1] == 'off':
            turn_off = turn_off + 1
    elif words[0] == 'Make':
        make = make + 1
    elif words[0] == 'Paint':
        paint = paint + 1

plt.figure(figsize=(7.5,5),dpi=80)
label = ['Paint', 'Put', 'Grasp', 'Move', 'Hide', 'Make', 'Throw', 'Turn on', 'Turn off', 'Open', 'Close']
sizes = np.array([paint, put, grasp, move, hide, make, throw, turn_on, turn_off, open, close])
explode = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
textprops = {'fontsize':16}
percent = 100.* sizes / sizes.sum()

patches, text = plt.pie(sizes, explode=explode, labels=label, labeldistance = 1.1, shadow = False, startangle = 90, textprops=textprops)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(label, percent)]
plt.legend(patches, labels, loc = 'best', fontsize=16)
# plt.tight_layout()
plt.axis('equal')
plt.show()



# All test goal distribution
throw = 0
open = 0
close = 0
grasp = 0
move = 0
put = 0
hide = 0
turn_on = 0
turn_off = 0
make = 0
paint = 0
for test in test_descriptions:
    words = test.split(' ')
    if words[0] == 'Throw':
        throw = throw + 1
    elif words[0] == 'Open':
        open = open + 1
    elif words[0] == 'Close':
        close = close + 1
    elif words[0] == 'Grasp':
        grasp = grasp + 1
    elif words[0] == 'Move':
        move = move + 1
    elif words[0] == 'Put':
        put = put + 1
    elif words[0] == 'Hide':
        hide = hide + 1
    elif words[0] == 'Turn':
        if words[1] == 'on':
            turn_on = turn_on + 1
        elif words[1] == 'off':
            turn_off = turn_off + 1
    elif words[0] == 'Make':
        make = make + 1
    elif words[0] == 'Paint':
        paint = paint + 1

plt.figure(figsize=(7.5,5),dpi=80)
label = ['Paint', 'Put', 'Hide', 'Grasp', 'Move', 'Make', 'Throw', 'Turn on', 'Turn off', 'Open', 'Close']
sizes = np.array([paint, put, hide, grasp, move, make, throw, turn_on, turn_off, open, close])
explode = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
textprops = {'fontsize':16}
percent = 100.* sizes / sizes.sum()

patches, text = plt.pie(sizes, explode=explode, labels=label, labeldistance = 1.1, shadow = False, startangle = 90, textprops=textprops)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(label, percent)]
plt.legend(patches, labels, loc = 'best', fontsize=16)
# plt.tight_layout()
plt.axis('equal')
plt.show()





# Train dataset descriptions distribution
throw = 0
open = 0
close = 0
grasp = 0
move = 0
put = 0
hide = 0
turn_on = 0
turn_off = 0
make = 0
paint = 0
nb_train_des = 0
for i in range(1344):
    if os.path.exists('./src/envs/dataset(goal)/train/' + str(i) + '/'):
        with np.load('./src/envs/dataset(goal)/train/' + str(i) + '/data.npz', allow_pickle=True) as data:
            descriptions = data['train_des']
        for d in descriptions:
            nb_train_des = nb_train_des + 1
            words = d.split(' ')
            if words[0] == "Throw":
                throw = throw + 1
            elif words[0] == "Open":
                open = open + 1
            elif words[0] == "Close":
                close = close + 1
            elif words[0] == "Grasp":
                grasp = grasp + 1
            elif words[0] == "Move":
                move = move + 1
            elif words[0] == "Put":
                put = put + 1
            elif words[0] == "Hide":
                hide = hide + 1
            elif words[0] == "Turn":
                if words[1] == 'on':
                    turn_on = turn_on + 1
                elif words[1] == 'off':
                    turn_off = turn_off + 1
            elif words[0] == "Make":
                make = make + 1
            elif words[0] == "Paint":
                paint = paint + 1

plt.figure(figsize=(7.5,5),dpi=80)
label = ['Put', 'Move', 'Paint', 'Throw', 'Make', 'Hide', 'Turn on', 'Grasp', 'Close', 'Open', 'Turn off']
sizes = np.array([put, move, paint, throw, make, hide, turn_on, grasp, close, open, turn_off])
explode = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
textprops = {'fontsize':16}
percent = 100.* sizes / sizes.sum()

patches, text = plt.pie(sizes, explode=explode, labels=label, labeldistance = 1.1, shadow = False, startangle = 90, textprops=textprops)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(label, percent)]
plt.legend(patches, labels, loc = 'best', fontsize=16)
# plt.tight_layout()
plt.axis('equal')
plt.show()



# Test dataset train descriptions distribution
throw = 0
open = 0
close = 0
grasp = 0
move = 0
put = 0
hide = 0
turn_on = 0
turn_off = 0
make = 0
paint = 0
nb_test_des_train = 0
for i in range(125):
    if os.path.exists('./src/envs/dataset(goal)/test/' + str(i) + '/'):
        with np.load('./src/envs/dataset(goal)/test/' + str(i) + '/data.npz', allow_pickle=True) as data:
            train_descriptions = data['train_des']
        for d in train_descriptions:
            nb_test_des_train = nb_test_des_train + 1
            words = d.split(' ')
            if words[0] == "Throw":
                throw = throw + 1
            elif words[0] == "Open":
                open = open + 1
            elif words[0] == "Close":
                close = close + 1
            elif words[0] == "Grasp":
                grasp = grasp + 1
            elif words[0] == "Move":
                move = move + 1
            elif words[0] == "Put":
                put = put + 1
            elif words[0] == "Hide":
                hide = hide + 1
            elif words[0] == "Turn":
                if words[1] == 'on':
                    turn_on = turn_on + 1
                elif words[1] == 'off':
                    turn_off = turn_off + 1
            elif words[0] == "Make":
                make = make + 1
            elif words[0] == "Paint":
                paint = paint + 1

plt.figure(figsize=(7.5,5),dpi=80)
label = ['Put', 'Move', 'Paint', 'Throw', 'Make', 'Hide', 'Turn on', 'Grasp', 'Close', 'Open', 'Turn off']
sizes = np.array([put, move, paint, throw, make, hide, turn_on, grasp, close, open, turn_off])
explode = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
textprops = {'fontsize':16}
percent = 100.* sizes / sizes.sum()

patches, text = plt.pie(sizes, explode=explode, labels=label, labeldistance = 1.1, shadow = False, startangle = 90, textprops=textprops)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(label, percent)]
plt.legend(patches, labels, loc = 'best', fontsize=16)
# plt.tight_layout()
plt.axis('equal')
plt.show()



# Test dataset test descriptions distribution
throw = 0
open = 0
close = 0
grasp = 0
move = 0
put = 0
hide = 0
turn_on = 0
turn_off = 0
make = 0
paint = 0
nb_test_des_test = 0
for i in range(125):
    if os.path.exists('./src/envs/dataset(goal)/test/' + str(i) + '/'):
        with np.load('./src/envs/dataset(goal)/test/' + str(i) + '/data.npz', allow_pickle=True) as data:
            test_descriptions = data['test_des']
        for d in test_descriptions:
            nb_test_des_test = nb_test_des_test + 1
            words = d.split(' ')
            if words[0] == "Throw":
                throw = throw + 1
            elif words[0] == "Open":
                open = open + 1
            elif words[0] == "Close":
                close = close + 1
            elif words[0] == "Grasp":
                grasp = grasp + 1
            elif words[0] == "Move":
                move = move + 1
            elif words[0] == "Put":
                put = put + 1
            elif words[0] == "Hide":
                hide = hide + 1
            elif words[0] == "Turn":
                if words[1] == 'on':
                    turn_on = turn_on + 1
                elif words[1] == 'off':
                    turn_off = turn_off + 1
            elif words[0] == "Make":
                make = make + 1
            elif words[0] == "Paint":
                paint = paint + 1

plt.figure(figsize=(7.5,5),dpi=80)
label = ['Put', 'Move', 'Paint', 'Hide', 'Grasp', 'Throw', 'Make', 'Turn on', 'Turn off', 'Open', 'Close']
sizes = np.array([put, move, paint, hide, grasp, throw, make, turn_on, turn_off, open, close])
explode = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
textprops = {'fontsize':16}
percent = 100.* sizes / sizes.sum()

patches, text = plt.pie(sizes, explode=explode, labels=label, labeldistance = 1.1, shadow = False, startangle = 90, textprops=textprops)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(label, percent)]
plt.legend(patches, labels, loc = 'best', fontsize=16)
# plt.tight_layout()
plt.axis('equal')
plt.show()




# Print the number of descriptions for 5 distributions above
# print('Number of all train descriptions: ', len(train_descriptions))
# print('Number of all test descriptions: ', len(test_descriptions))
print('Number of train dataset train descriptions: ', nb_train_des)
print('Number of test dataset train descriptions: ', nb_test_des_train)
print('Number of test dataset test descriptions: ', nb_test_des_test)