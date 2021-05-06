from src.envs.color_generation import sample_color
from src.envs.env_params import get_env_params
import numpy as np
import pybullet as p
import os
import pickle

ENV_PARAMS = get_env_params()

path = "./shapenet_objects/"
objects = [o for o in os.listdir(path) if 'urdf' in o and '_prototype' not in o]
with open(path + 'sizes.pkl', 'rb') as f:
    object_sizes = pickle.load(f)




class Thing:
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):

        assert color in env_params['colors_attributes']

        self.env_params = env_params
        self.id = object_id
        self.bullet_client = bullet_client
        self.type = object_type
        self.get_type_encoding(object_type)
        self.color = color
        self.render_mode = False  # render_mode
        self.attributes = []
        if 'categories' in self.env_params['admissible_attributes']:
            self.attributes += ['thing', object_type, self.color]
        self.categories = []
        self.features = []

        self.type_encoding = None
        self.position = None
        self.orientation = None
        self.size_encoding = None
        self.rgb_encoding = None

        self.objects = None

        self.touched = False
        self.grasped = False

        self.sample_size()
        self.sample_color()
        self.initial_rgb_encoding = self.rgb_encoding.copy()
        self.p_id = self.generate_object()
        self.sample_position(objects)
        self.update_color()



    def give_ref_to_obj_list(self, objects):
        self.objects = objects

    def sample_color(self):
        self.rgb_encoding = np.array(list(sample_color(color=self.color)) + [1])


    def update_attributes(self):
        if 'absolute_location' in self.env_params['admissible_attributes']:
            self.update_absolute_location_attributes(self.position)
        if 'color' in self.env_params['admissible_attributes']:
            self.update_color_attributes()

    def update_all_attributes(self):
        self.update_attributes()

    def update_color_attributes(self, old_color=None):
        if self.color not in self.attributes:
            self.attributes.append(self.color)
            if old_color is not None:
                self.attributes.remove(old_color)

    def update_absolute_location_attributes(self, new_position):
        # update absolute geo_location of objects
        if new_position[0] < 0:
            self.attributes[0] = 'left'
        else:
            self.attributes[0] = 'right'

        if new_position[1] < 0:
            self.attributes[1] = 'bottom'
        else:
            self.attributes[1] = 'top'

   
    def update_position(self, new_position=None):
        if new_position is None:
            new_position, new_orientation = self.bullet_client.getBasePositionAndOrientation(self.p_id)
            self.orientation = np.array(new_orientation).copy()
            new_position = np.array(new_position)

        if 'absolute_location' in self.env_params['admissible_attributes']:
            self.update_absolute_location_attributes(new_position)

        # update relative attributes
        self.position = new_position.copy()

    def get_type_encoding(self, object_type):
        self.type_encoding = np.zeros([self.env_params['nb_types']])
        self.type_encoding[self.env_params['types'].index(object_type)] = 1


    def get_features(self):
        self.features = dict(pos=self.position,
                             orn=self.orientation,
                             # vel=self.velocity,
                             color=self.rgb_encoding[:3],
                             size=np.array([self.size_encoding])
                             )
        return self.features.copy()

    def compute_radius(self):
        return np.sqrt((self.sizes[0]/2)**2 + (self.sizes[1]/2)**2)

    def sample_position(self, objects):
        ok = False
        while not ok:
            is_left = np.random.rand() < 0.5
            low, high = np.array(self.env_params['table_ranges'])[:, 0], np.array(self.env_params['table_ranges'])[:, 1]
            if is_left:
                print('left')
                high[0] = -0.19
            else:
                print('right')
                low[0] = 0.19
            candidate_position = np.random.uniform(low=low, high=high)
            candidate_position = np.array(list(candidate_position) + [-0.025 + self.sizes[2] + 0.0005])
            ok = True
            for obj in objects:
                if np.linalg.norm(obj.position - candidate_position) < (self.compute_radius() + obj.compute_radius())*1.3:
                    ok = False
            if ok:
                # set object in correct position
                orientation = np.random.uniform(-180, 180)
                self.bullet_client.resetBasePositionAndOrientation(self.p_id, candidate_position, self.bullet_client.getQuaternionFromEuler(np.deg2rad([0, 0, orientation])))
                #[0.0, 0.0, 0.7071, 0.7071])
                # update position encoding
                self.update_position()

    def sample_size(self):
        self.size_encoding = np.random.uniform(self.env_params['min_max_sizes'][0], self.env_params['min_max_sizes'][1])

    def update_color(self, new_color=None, new_rgb=None):
        if new_color is not None:
            old_color = self.color
            self.color = new_color
            self.rgb_encoding = new_rgb
            self.update_color_attributes(old_color)
        self.bullet_client.changeVisualShape(self.p_id, -1, rgbaColor=self.rgb_encoding)

    def __repr__(self):
        return 'Object # {}: {} {}'.format(self.id, self.color, self.type)




# SHAPENET objects

class ShapeNet(Thing):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)

    def scan_objects_and_save_their_sizes(self):
        # MAKE SURE to scan max 10 objects at a time, otherwise you'll crash your computer
        objects = sorted([o for o in os.listdir(path) if 'urdf' in o and '_prototype' not in o])

        if os.path.exists(path + 'sizes.pkl'):
            with open(path + 'sizes.pkl', 'rb') as f:
                max_sizes = pickle.load(f)
        else:
            max_sizes = dict()
        for o in objects[:10]: ## CHANGE INDS HERE TO SAVE SOME OBJECTS
            print(o)
            object_urdf = path + o
            boxStartOr = p.getQuaternionFromEuler(np.deg2rad([0, 0, 0]))
            boxId = p.loadURDF(object_urdf, [0, 0, 0], boxStartOr)
            sizes = np.array(p.getAABB(boxId)[1]) - np.array(p.getAABB(boxId)[0])
            max_sizes[o] = sizes
        with open(path + 'sizes.pkl', 'wb') as f:
            pickle.dump(max_sizes, f)

    def generate_object(self):
        # MAKE SURE to scan max 10 objects at a time, otherwise you'll crash your computer
        # self.scan_objects_and_save_their_sizes()

        # code for now, until we load the urdf models and save their sizes in the sizes.pkl file for all objects listed in env_params
        objects = sorted([o for o in os.listdir(path) if 'urdf' in o and '_prototype' not in o])
        o = np.random.choice(objects)

        # after
        # o = self.type

        object_urdf = path + o
        original_sizes = object_sizes[o]  # get original size
        ratio = self.size_encoding / np.sqrt(np.sum(np.array(original_sizes)**2)) # compute scaling ratio
        boxStartOr = self.bullet_client.getQuaternionFromEuler(np.deg2rad([0, 0, 0]))
        boxId = self.bullet_client.loadURDF(object_urdf, [0, 0, 0], boxStartOr, globalScaling=ratio)
        self.sizes = original_sizes.copy() * ratio
        # self.sizes = np.array(p.getAABB(boxId)[1]) - np.array(p.getAABB(boxId)[0])
        self.bullet_client.changeDynamics(boxId, -1, linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001)
        self.bullet_client.changeVisualShape(boxId, -1, rgbaColor=self.rgb_encoding + [1])
        return boxId

 


# CATEGORIES
            
class Solid(Thing):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)
        if 'category' in env_params['admissible_attributes']:
            self.attributes += ['solid']

class Animal(ShapeNet):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)
        if 'category' in env_params['admissible_attributes']:
            self.attributes += ['animal']

class Food(ShapeNet):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)
        if 'category' in env_params['admissible_attributes']:
            self.attributes += ['food']

class Kitchenware(ShapeNet):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)
        if 'category' in env_params['admissible_attributes']:
            self.attributes += ['kitchenware']

class Vehicle(ShapeNet):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)
        if 'category' in env_params['admissible_attributes']:
            self.attributes += ['vehicle']


# OBJECTS
class Cube(Solid):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)
            
    def generate_object(self):
        sizes = [self.size_encoding / (np.sqrt(3) * 2) * 0.75] * 3
        self.sizes = sizes.copy()

        colcubeId = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_BOX, halfExtents=sizes)
        visplaneId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_BOX, halfExtents=sizes, rgbaColor=list(self.rgb_encoding))
        legoUID = self.bullet_client.createMultiBody(0.3, colcubeId, visplaneId, self.position)
        self.bullet_client.changeDynamics(legoUID, -1, lateralFriction=1.5)
        return legoUID


class Block(Solid):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)

    def generate_object(self):
        sizes = [self.size_encoding / np.sqrt(6)] + [self.size_encoding / (np.sqrt(6) * 2)] * 2
        self.sizes = sizes.copy()
        self.sizes = sizes.copy()

        colcubeId = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_BOX, halfExtents=sizes)
        visplaneId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_BOX, halfExtents=sizes, rgbaColor=list(self.rgb_encoding))
        legoUID = self.bullet_client.createMultiBody(0.3, colcubeId, visplaneId, self.position)
        self.bullet_client.changeDynamics(legoUID, -1, lateralFriction=1.5)
        return legoUID


class Cylinder(Solid):
    def __init__(self, env_params, bullet_client, object_type, color, object_id, objects):
        super().__init__(env_params, bullet_client, object_type, color, object_id, objects)

    def generate_object(self):
        #TODO: update to generate a cylinder here
        sizes = [self.size_encoding / np.sqrt(6)] + [self.size_encoding / (np.sqrt(6) * 2)] * 2
        self.sizes = sizes.copy()

        colcubeId = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_BOX, halfExtents=sizes)
        visplaneId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_BOX, halfExtents=sizes, rgbaColor=list(self.rgb_encoding))
        legoUID = self.bullet_client.createMultiBody(0.3, colcubeId, visplaneId, self.position)
        self.bullet_client.changeDynamics(legoUID, -1, lateralFriction=1.5)
        return legoUID


# build a dict of the classes
things_classes = dict(cube=Cube,
                      block=Block,
                      cylinder=Cylinder)
shapenet_categories = list(ENV_PARAMS['attributes']['categories'])
shapenet_categories.remove('solid')
shapenet_obj = []
cat_dict = dict(kitchenware=Kitchenware,
                vehicle=Vehicle,
                food=Food,
                animal=Animal)
for c in shapenet_categories:
    things_classes.update(dict(zip(ENV_PARAMS['categories'][c], [cat_dict[c]] * len(ENV_PARAMS['categories'][c]))))


def build_object(env_params, bullet_client, object_type, color, object_id, objects):
    assert object_type in env_params['types']
    obj_class = things_classes[object_type](env_params, bullet_client, object_type, color, object_id, objects)
    assert obj_class.type == object_type, '{}, {}'.format(obj_class.type, object_type)
    return obj_class

stop = 1