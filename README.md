# A New Environment for Intrinsically Motivated Goal-Conditioned Language Reinforcement Learning
![MyEnv](https://user-images.githubusercontent.com/59849265/139904772-1afef311-5d54-4685-a972-f1d7837c3c3e.png)

## Introduction

This is the latest version of the environment. It is built in [PyBullet](https://pybullet.org/wordpress/) and can be teleoperated by **VR device**. As shown in the picture above, this environment contains a [UNIVERSAL ROBOT UR5e](https://www.universal-robots.com/products/ur5-robot/) robot arm with a gripper. We need to use VR device to teleoperate the robot arm for the collection of expert data. This environment also contains a table with a shelf, a drawer, a door, a panel, and three light bulbs with three corresponding buttons.

More specifically, the environment contains:

* **The shelf**: We can put objects on the shelf or under the shelf with the door closed in order to hide objects.
* **The drawer**: We can open the drawer in order to take out objects or to put objects into the drawer. We can also close the drawer in order to hide objects or just to close it.
* **The door**: The door can be moved along one axis. We define that opening the door means moving the door to the left side of the shelf, and closing the door means moving the door to the right side of the shelf.
* **Three light bulbs with three buttons**: Three light bulbs are installed on the upper side of the shelf and three corresponding buttons are installed in the left side of the shelf. Pushing down the button makes the light bulb light up or light off. When the light is off, the light bulb is white. When the light is on, the colors of the light bulbs are red, green, blue.
* **The panel**: The panel has two functions. Firstly, it shows the mixed color from the light bulbs which are on. There are 8 combinations of colors: red, green, blue, yellow, magenta, cyan, white, black. When there is only one light on, the panel shows the color of that light. Yellow is the combination of red and green. Magenta is the combination of red and blue. Cyan is the combination of green and blue. White is the combination of red, green and blue. When all lights are off, the panel is black. Secondly, the panel can paint the contacted object into the the current color of the panel.
* **The objects**: The objects are classified in five categories: geometric solid, kitchen ware, animal model, food model, vehicle model. The objects of geometric solid are designed with [PyBullet](https://pybullet.org/wordpress/) and [Blender](https://www.blender.org/). Other objects are selected from [ShapeNet](https://shapenet.org/). The names of all objects are shown in the “Grammar” section below. When we generate the initial scene from the description, the environment just randomly selects 3 objects in the initial scene. However, there must be the objects which make the goals achievable.

## Grammar

*{…} represents the set of words or phrases, and others represent the specific words. U represent “OR” among multiple possible choices.*

**Throw**:
* <throw + {max_nb_objects} + objects on the floor>

**Open**:
* <open + the + {drawer/door}>

**Close**:
* <close + the + {drawer/door}>

**Grasp**:
* <grasp + any + {color} + object>
* <grasp + {color} U any + {object} U {object category}>

**Move**:
* <move + any + {color} + object>
* <move + {color} U any + {object} U {object category}>

**Put**:
* <put + {any/all} + {color} + object + {position}>
* <put + {color} U {any/all} + {object} U {object category} + {position}>

**Hide**:
* <hide + {any/all} + {color} + object>
* <hide + {color} U {any/all} + {object} U {object category}>

**Turn on**:
* <turn on + the + {rgbb} + light>

**Turn off**:
* <turn off + the + {rgbb} + light>

**Make**:
* <make + the + panel + {color}>

**Paint**:
* <paint + {any/all} + {color1} + object + {color2}>
* <paint + {color1} + {object} U {object category} + {color2}>
* <paint + {any/all} + {object} U {object category} + {color}>

*The sets of words or phrases are explained below.*

**{max_nb_objects}**: 0, 1, ..., max_nb_objects-1

**{drawer/door}**: drawer, door

**{any/all}**: any, all

**{position}**: on the left side of the table, on the right side of the table, on the shelf, behind the door, in the drawer

**{rgbb}**: red, green, blue

**{color1} should be different from {color2}**

**{color}**: red, green, blue, yellow, magenta, cyan, white, black

**{object category}**: solid, kitchenware, animal, food, vehicle

**{object}**:

* **solid:**  cube, block, cylinder
* **kitchenware:**  bottle, bowl, plate, cup, spoon
* **animal:**  bear, bird, dog, fish, elephant
* **food:**  apple, banana, cookie, donut, sandwich
* **vehicle:**  train, plane, car, bike, bus

## Generalization in the Testing Set
Like [IMAGINE](https://arxiv.org/abs/2002.09253) approach, agents can only benefit from goal imagination when their reward function is able to generalize the meanings of imagined goals from the meanings of known ones. When it works, agents can further train on imagined goals, which might reinforce the generalization of the policy. This section introduces different types of generalizations that the reward and policy can both demonstrate. The following goals are removed from the training set and added to the testing set.

* **Type 1 - Attribute-Object Generalization:** In order to measure the ability to accurately associate an attribute and an object that were never seen together before, we removed from the training set all goals containing the following attribute-object combinations: **{'red bear', 'green donut', 'blue bowl', 'white cube', 'black car'}** and added them to the testing set.
* **Type 2 - Object Identification:** In order to measure the ability to identify a new object from its attribute, we removed from the training set all goals containing the object: **{'bird'}** and added them to the testing set.
* **Type 3 - Object-Position Generalization:** In order to measure the ability to accurately associate an object and a position that were never seen together before, we removed from the training set all goals containing the following object-position combinations: **{'apple on the right side of the table'}** and added them to the testing set.
* **Type 4 - Predicate-Category Generalization:** In order to measure the ability to accurately associate a predicate and a category that were never seen together before, we removed from the training set all goals containing **'grasp' predicate and 'food' category** and added them to the testing set.
* **Type 5 - Predicate-Object Generalization:** In order to measure the ability to accurately associate a predicate and an object that were never seen together before, we removed from the training set all goals containing **'move' predicate and 'bottle' object** and added them to the testing set.
* **Type 6 - Predicate dynamics generalization:** 'Hide' action can lead to two possible results with same object and same attribute. Both *hiding the object in the drawer* and *hiding the object behind the door* can achieve the 'hide' goal. Therefore, we would like to see how agent will hide a category of objects which it never hides during the training. In order to measure the ability to accurately associate a predicate and a category of objects that were never seen together before, we removed from the training set all goals containing **'hide' predicate and all objects from 'vehicle' category** and added them to the testing set.
* **Type 7 - Attribute-Position Generalization:** In order to measure the ability to accurately associate an attribute and a position that were never seen together before, we removed from the training set all goals containing **'yellow' attribute and 'in the drawer' position** and added them to the testing set. In fact, only 'put' predicate has this attribute-position combination.
* **Type 8 - Attribute-Attribute Generalization:** In order to measure the ability to accurately associate an attribute (color) and another attribute (color) that were never seen together before, we removed from the training set all goals containing **'black' original attribute and 'magenta' changed attribute** and added them to the testing set. In fact, only 'paint' predicate has this attribute (color)-attribute (color) combination.

## Goal Distribution
According to the grammar of this environment and the generalization in the testing set, there are 4462 goals in total, including 4107 training goals and 355 test goals.

## VR Setup

1. (Example of HTC Vive and Windows10) Set up the VR headset, base stations, controllers. Install SteamVR on PC.
2. Follow the steps in https://docs.google.com/document/d/1I4m0Letbkw4je5uIBxuCfhBcllnwKojJAyYSTjHbrH8/edit
3. Download or git clone https://github.com/bulletphysics/bullet3
4. Make sure to have Microsoft Visual Studio (later than VS2010) on your PC. 
5. In the bullet3, click on *build_visual_studio_vr_pybullet_double.bat* and open *build3/vs2010/0_Bullet3Solution.sln* . When asked, convert the projects to a newer version of Visual Studio. If you installed Python in the C:\ root directory, the batch file should find it automatically. Otherwise, edit this batch file to choose where Python include/lib directories are located.
6. Make sure to switch to 'Release' and 'x64' configuration of Microsoft Visual Studio and build the project. 
7. In the bin folder of bullet3, run *App_PhysicsServer_SharedMemory_VR_vs2010_x64_release.exe*
8. Run *vr_data_collection.py* in our environment folder in order to collect the trajectories with the VR set.


## VR instructions for the demonstrator

1. After running the program, you are asked to type your name in the terminal.
2.	Stand between two base stations. 
3.	Wear the VR headset and hold two controllers in both hands. 
4.	Press the trackpad of the left controller to start/finish saving. When you start saving, you will see the current goal in white. Below the goal, you will see “Start saving: Goal {X}” in green. {X} is the number of goals which you have recorded. When you finish saving, you will see “Finish saving: Goal {X}” in blue. 
5.	Press the trackpad of the right controller to cancel saving. When you cancel saving, you will see “Cancel saving: Goal {X}” in red. 
6.	Press the trigger of the right controller if you want to close the gripper of robot arm during the saving process. The gripper always tracks the right controller.
7.	When the goal begins with “Grasp”, make sure to finish saving while grasping the required object with gripper.
8.	We define that “Open the door” means that moving the door to the left and “Close the door” means that moving the door to the right. 
9.	When the goal begins with “Move”, do not move the required object on the panel. This may change the color of the object, which is not the goal.
10.	When the goal begins with “Hide” or “Put” something “behind the door”, close the door finally.
11.	Push the button in the left side of shelf to turn on/off red, green, blue light. 
12.	The panel has two functions. Firstly, it shows the mixed color from the light bulbs which are on. There are 8 combinations of colors: red, green, blue, yellow, magenta, cyan, white, black. Yellow is the combination of red and green. Magenta is the combination of red and blue. Cyan is the combination of green and blue. White is the combination of red, green and blue. When all lights are off, the panel is black. Secondly, the panel can paint the contacted object into current color of the panel.

![Controller](https://user-images.githubusercontent.com/59849265/139906630-8463522d-e91b-4e65-a1b9-0749c6d11764.png)

## Code Usage
In order to run the code, the current path in the terminal should be just inside the captionRL-env repository.
* ```python -m src.envs.analyze_dataset```: Dataset analysis in pie chart of all predicates distribution.
* ```python -m src.envs.interactive```: Rendering the environment and interact with it on PC.
* ```python -m src.envs.utils```: By commenting and uncommenting the code blocks, you can: 1. reconstruct the initial and final image of one episode/trajectory; 2. check if the reward of each episode/trajectory is "True"; 3. create the train and the test dataset. You also need to change the path in the code, and change/define your own criteria of separating the train and test dataset if needed.
* ```python -m src.envs.vr_data_collection```: Data/Trajectories collection with VR.

**Important**: In the "reset" function of *instance.py*: 1. if you run *interactive.py*, please comment line 333 and line 351; 2. if you regenerate the scene/image, please comment line 349 and line 351.\

The environment is inspired by:\
[1] Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, and Pierre Sermanet. Learning latent plans from play. Conference on Robot Learning (CoRL), 2019.\
[2] Cédric Colas, Tristan Karch, Nicolas Lair, Jean-Michel Dussoux, Clément Moulin-Frier, Peter Dominey, and Pierre-Yves Oudeyer. Language as a cognitive tool to imagine goals in curiosity driven exploration. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 3761–3774. Curran Associates, Inc., 2020.\
[3] Transfer Learning from Play and Language - Nailing the Baseline: https://sholtodouglas.github.io/Learning-from-Play/

