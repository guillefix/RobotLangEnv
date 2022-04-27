
import pickle
root_folder="/home/guillefix/code/inria/captionRLenv/"
types = pickle.load(open(root_folder+"object_types.pkl","rb"))
geometric_solid = ('cube', 'block', 'cylinder')
kitchen_ware = ('bottle', 'bowl', 'plate', 'cup', 'spoon')
animal_model = ('bear', 'bird', 'dog', 'fish', 'elephant')
food_model = ('apple', 'banana', 'cookie', 'donut', 'sandwich')
vehicles_model = ('train', 'plane', 'car', 'bike', 'bus')

categories = dict(solid = geometric_solid,
                  kitchenware = kitchen_ware,
                  animal = animal_model,
                  food = food_model,
                  vehicle = vehicles_model,
                  )

types

colors = list(('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'black'))
colors = tuple(colors)
positions = ('on the left side of the table', 'on the right side of the table', 'on the shelf', 'behind the door', 'in the drawer')
drawer_door = ('drawer', 'door')
any_all = ('any', 'all')
rgbb = ('red', 'green', 'blue')

attributes = dict(types=types,
                  categories=tuple(categories.keys()),
                  colors=colors,
                  positions=positions,
                  drawer_door=drawer_door,
                  any_all=any_all,
                  rgbb=rgbb)
admissible_actions=('Open', 'Close', 'Grasp', 'Put', 'Hide', 'Turn on', 'Turn off', 'Make', 'Paint', 'Move', 'Throw')

extra_words = ['Throw objects on the floor', 'Open the', 'Close the', 'Grasp any object', 'Grasp', 'Move any object', 'Move', 'Put object', 'Put', 'Hide object', 'Hide', 'Turn on the light', 'Turn off the light', 'Make the panel', 'Paint object', 'Paint', 'Paint']

words = sum([sum([v2.split(" ") for v2 in v], []) for k,v in attributes.items()], []) + list(admissible_actions) + sum([w.split(" ") for w in extra_words], []) + ["0","1","2","3"]

import numpy as np


unique_words = np.unique(words)

unique_words

vocab_dict = {x:str(i) for i,x in enumerate(unique_words)}
vocab_dict_reverse = {str(i):x for i,x in enumerate(unique_words)}
len(vocab_dict)
#%%

# from constants import *
root_dir="/home/guillefix/code/inria/"
import json
with open(root_dir+"UR5_processed/npz.annotation.txt.annotation.class_index.json", "w") as f:
    f.write(json.dumps(vocab_dict))
with open(root_dir+"UR5_processed/npz.annotation.txt.annotation.class_index_reverse.json", "w") as f:
    f.write(json.dumps(vocab_dict_reverse))
