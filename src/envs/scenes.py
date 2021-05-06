import numpy as np
import os
from src.envs.objects import build_object

TEST_OLD = True

color_dict = dict(red=[1, 0, 0], green=[0, 1, 0], blue=[0, 0, 1], magenta=[1, 0, 1], yellow=[1, 1, 0], cyan=[0, 1, 1], black=[0, 0, 0], white=[1, 1, 1])
rgb_dict = dict(zip([str([int(c) for c in v]) for v in list(color_dict.values())], list(color_dict.keys())))


def complex_scene(bullet_client, env_params, offset, flags, env_range_low, env_range_high, num_objects, description=None):

    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent, plane_extent, 0.0001])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, plane_extent, 0.0001],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, 0, -0.27])


    door = add_door(bullet_client)
    drawer = add_drawer(bullet_client)
    pad = add_pad(bullet_client)
    button_red, toggleSphere_red = add_button(bullet_client, position=(-0.48, 0.45), color=(1, 0, 0))
    button_green, toggleSphere_green = add_button(bullet_client, position=(-0.38, 0.45), color=(0, 1, 0))
    button_blue, toggleSphere_blue = add_button(bullet_client, position=(-0.28, 0.45), color=(0, 0, 1))

    add_static(bullet_client)

    # make objects
    legos, legos_p_ids = sample_objects(description, bullet_client, env_params, num_objects)

    return door, drawer, pad, legos, legos_p_ids, [door,button_red, button_green, button_blue], {button_red: ('button_red', toggleSphere_red), button_green: ('button_green',
                                                                                                                                                 toggleSphere_green),
                                                                         button_blue: ('button_blue', toggleSphere_blue)} # return the toggle sphere with it's joint index


def sample_objects(description, bullet_client, env_params, num_objects):
    objects_to_add = get_required_obj(description)
    legos = get_objects(env_params, bullet_client, objects_to_add, num_objects)[0]
    legos_p_ids = [l.p_id for l in legos]
    return legos, legos_p_ids

def get_required_obj(description):
    objects_to_add = []
    return objects_to_add

def get_obj_identifier(env_params, object_type, color):
    type_id = str(env_params['types'].index(object_type))
    if len(type_id) == 1:
        type_id = '0' + type_id

    color_id = str(env_params['colors_attributes'].index(color))
    return type_id + color_id


def get_objects(env_params, bullet_client, objects_to_add, num_objects):
    objects = []
    objects_ids = []
    objects_types = []
    if objects_to_add is not None:
        for object in objects_to_add:
            if object['type'] is not None:
                type = object['type']
            elif object['category'] is not None:
                type = np.random.choice(env_params['categories'][object['category']])
            else:
                type = np.random.choice(env_params['types'])
            if object['color'] is not None:
                color = object['color']
            else:
                color = np.random.choice(env_params['colors_attributes'])

            obj_id = get_obj_identifier(env_params, type, color)
            if obj_id not in objects_ids:
                objects.append(build_object(env_params, bullet_client, type, color, len(objects), objects))
                objects_ids.append(obj_id)
                objects_types.append(type)

    while len(objects) < num_objects:
        type = np.random.choice(env_params['types'])
        color = np.random.choice(env_params['colors_attributes'])
        obj_id = get_obj_identifier(env_params, type, color)
        if obj_id not in objects_ids:
            objects.append(build_object(env_params, bullet_client, type, color, len(objects), objects))
            objects_ids.append(obj_id)
            objects_types.append(type)
    inds = list(range(len(objects)))
    np.random.shuffle(inds)
    objects = [objects[i] for i in inds]
    objects_ids = [objects_ids[i] for i in inds]
    objects_types = [objects_types[i] for i in inds]
    print(objects_types)
    return objects, objects_ids, objects_types


def add_static(bullet_client):
    texUid = bullet_client.loadTexture(os.path.dirname(os.path.abspath(__file__)) + '/env_meshes/wood.png')

    def create(halfExtents, location):
        colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=halfExtents)
        if TEST_OLD:
            visplaneId =  bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=halfExtents,
                                                 rgbaColor=[0.75, 0.4, 0.2, 1])
            block = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, location)

        else:
            visplaneId =  bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=halfExtents)
            block = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, location)
            bullet_client.changeVisualShape(block, -1, textureUniqueId=texUid)

    # TableTop
    width = 0.6
    create(halfExtents=[width, 0.28, 0.005], location = [0, 0.25, -0.03])
    # Cabinet back
    create(halfExtents=[width, 0.01, 0.235], location = [0., 0.52, -0.00])
    # Cabinet top
    width = 0.62
    create(halfExtents=[width, 0.065, 0.005], location =[0., 0.45, 0.24])
    # Cabinet sides
    width = 0.03
    create(halfExtents=[width, 0.065, 0.235], location =[-0.59, 0.45, -0.00])
    create(halfExtents=[width, 0.065, 0.235], location =[0.59, 0.45, -0.00])


def add_door(bullet_client, offset=np.array([0, 0, 0]), flags=None, ghostly=False):
    sphereRadius = 0.1
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

    #     wallid = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
    #                                       halfExtents=[sphereRadius*4, sphereRadius/4, sphereRadius*4])
    wallid = bullet_client.createCollisionShape(bullet_client.GEOM_MESH, fileName=os.path.dirname(os.path.abspath(__file__)) + '/env_meshes/door2.obj', meshScale=[0.0015] * 3,
                                        flags=bullet_client.GEOM_FORCE_CONCAVE_TRIMESH)

    mass = 0
    visualShapeId = -1

    link_Masses = [0.1]
    linkCollisionShapeIndices = [wallid]
    if ghostly:
        visId = bullet_client.createVisualShape(bullet_client.GEOM_MESH, fileName = os.path.dirname(os.path.abspath(__file__)) + '/env_meshes/door_textured.obj', meshScale = [0.0015] * 3,  flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH, rgbaColor=[0,0,1,0.5])
        linkVisualShapeIndices = [visId]
    else:
        visId = bullet_client.createVisualShape(bullet_client.GEOM_MESH, fileName = os.path.dirname(os.path.abspath(__file__)) + '/env_meshes/door_textured.obj',
                                                meshScale = [0.0015] * 3,  flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH, rgbaColor=[0,1,1,0.5])
        if TEST_OLD:
               linkVisualShapeIndices = [-1]
        else:
                linkVisualShapeIndices = [visId]

    linkPositions = [[0.0, 0.0, 0.27]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])]
    linkInertialFramePositions = [[0, 0, 0.0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    # jointTypes = [bullet_client.JOINT_REVOLUTE]
    jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([0, 0.4, -0.2]) + offset
    baseOrientation = [0, 0, 0, 1]

    sphereUid = bullet_client.createMultiBody(mass,
                                              colBoxId,
                                              visualShapeId,
                                              basePosition,
                                              baseOrientation,
                                              linkMasses=link_Masses,
                                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                                              linkVisualShapeIndices=linkVisualShapeIndices,
                                              linkPositions=linkPositions,
                                              linkOrientations=linkOrientations,
                                              linkInertialFramePositions=linkInertialFramePositions,
                                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                                              linkParentIndices=indices,
                                              linkJointTypes=jointTypes,
                                              linkJointAxis=axis)

    bullet_client.changeDynamics(sphereUid,
                                 -1,
                                 spinningFriction=0.001,
                                 rollingFriction=0.001,
                                 linearDamping=0.0)
    if ghostly:
        collisionFilterGroup = 0
        collisionFilterMask = 0
        bullet_client.setCollisionFilterGroupMask(sphereUid, -1, collisionFilterGroup, collisionFilterMask)
        for i in range(0, bullet_client.getNumJoints(sphereUid)):
            bullet_client.setCollisionFilterGroupMask(sphereUid, i, collisionFilterGroup,
                                                           collisionFilterMask)
    return sphereUid


def add_button(bullet_client, offset=np.array([0, 0, 0]), position=(-0.48, 0.45), color=(0,0,0), ghostly = False):
    sphereRadius = 0.02
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius / 4])

    mass = 0
    if ghostly:
        rgbaColor = list(color) + [0.5]
        # rgbaColor = [1,0,0,0.5]
    else:
        rgbaColor = list(color) + [1]
        # rgbaColor = [1,0,0,1]

    visualShapeId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                        halfExtents=[sphereRadius, sphereRadius, sphereRadius / 4],
                                                        rgbaColor=rgbaColor)


    link_Masses = [0.1]
    linkCollisionShapeIndices = [colBoxId]
    linkVisualShapeIndices = [visualShapeId]
    x = position[0]
    y = position[1]
    linkPositions = [[x, y, 0.8]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([0, 0, 0])]
    linkInertialFramePositions = [[0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    # jointTypes = [bullet_client.JOINT_REVOLUTE]
    jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([0, 0, -0.7]) + offset
    baseOrientation = [0, 0, 0, 1]

    sphereUid = bullet_client.createMultiBody(mass,
                                              colBoxId,
                                              visualShapeId,
                                              basePosition,
                                              baseOrientation,
                                              linkMasses=link_Masses,
                                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                                              linkVisualShapeIndices=linkVisualShapeIndices,
                                              linkPositions=linkPositions,
                                              linkOrientations=linkOrientations,
                                              linkInertialFramePositions=linkInertialFramePositions,
                                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                                              linkParentIndices=indices,
                                              linkJointTypes=jointTypes,
                                              linkJointAxis=axis)

    bullet_client.changeDynamics(sphereUid,
                                 -1,
                                 spinningFriction=0.001,
                                 rollingFriction=0.001,
                                 linearDamping=0.0)
    bullet_client.setJointMotorControl2(sphereUid, 0, bullet_client.POSITION_CONTROL, targetPosition=0.03, force=1)


    if ghostly:
        collisionFilterGroup = 0
        collisionFilterMask = 0
        bullet_client.setCollisionFilterGroupMask(sphereUid, -1, collisionFilterGroup, collisionFilterMask)
        for i in range(0, bullet_client.getNumJoints(sphereUid)):
            bullet_client.setCollisionFilterGroupMask(sphereUid, i, collisionFilterGroup,
                                                           collisionFilterMask)
        toggleSphere = None
    else:
        # create a little globe to turn on and off
        sphereRadius = 0.03
        colSphereId = bullet_client.createCollisionShape(bullet_client.GEOM_SPHERE, radius=sphereRadius)
        visualShapeId = bullet_client.createVisualShape(bullet_client.GEOM_SPHERE,
                                                        radius=sphereRadius,
                                                        rgbaColor=[1, 1, 1, 1])
        toggleSphere = bullet_client.createMultiBody(0.0, colSphereId, visualShapeId, [x, y, 0.24],
                                                     baseOrientation)

    return sphereUid, toggleSphere


def add_drawer(bullet_client, offset=np.array([0, 0, 0]), flags=None, ghostly=False):
    texUid = bullet_client.loadTexture(os.path.dirname(
        os.path.abspath(__file__)) + '/env_meshes/wood.png')

    # if not ghostly:
    #     # add in the blockers to prevent it being pulled all the way out
    #     half_extents = [0.1, 0.28, 0.005]
    #     colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
    #     visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents,
    #                                                  rgbaColor=[0.75, 0.4, 0.2, 1])
    #     bottom = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.13, 0.25, -0.13])
    #     bullet_client.changeVisualShape(bottom, -1, textureUniqueId=texUid)

    #     half_extents = [0.1, 0.05, 0.015]
    #     colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
    #     visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents,
    #                                                  rgbaColor=[0.75, 0.4, 0.2, 1])
    #     back = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [0, 0.25, -0.06])
    #     bullet_client.changeVisualShape(back, -1, textureUniqueId=texUid)
    #     half_extents = [0.03, 0.01, 0.045]
    #     colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
    #     visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents,
    #                                                  rgbaColor=[0.75, 0.4, 0.2, 1])
    #     side1 = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.25, -0.02, -0.08])
    #     bullet_client.changeVisualShape(side1, -1, textureUniqueId=texUid)

    #     colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
    #     visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents,
    #                                                  rgbaColor=[0.75, 0.4, 0.2, 1])
    #     side2 = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.0, -0.02, -0.08])
    #     bullet_client.changeVisualShape(side2, -1, textureUniqueId=texUid)

    if not ghostly:
        # add in the blockers to prevent it being pulled all the way out
        half_extents = [0.24, 0.28, 0.005]
        colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        bottom = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.265+0.265, 0.25, -0.13])
        bullet_client.changeVisualShape(bottom, -1, textureUniqueId=texUid)

        half_extents = [0.22, 0.05, 0.015]
        colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        back = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.22+0.265, 0.25, -0.06])
        bullet_client.changeVisualShape(back, -1, textureUniqueId=texUid)
        half_extents = [0.03, 0.01, 0.045]
        colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        side1 = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.527+0.265, -0.02, -0.08])
        bullet_client.changeVisualShape(side1, -1, textureUniqueId=texUid)

        colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=half_extents)
        side2 = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.0+0.265, -0.02, -0.08])
        bullet_client.changeVisualShape(side2, -1, textureUniqueId=texUid)


    wallid = bullet_client.createCollisionShape(bullet_client.GEOM_MESH, fileName=os.path.dirname(
        os.path.abspath(__file__)) + '/env_meshes/drawer6.obj', meshScale=[1.25] * 3,
                                                flags=bullet_client.GEOM_FORCE_CONCAVE_TRIMESH)

    if ghostly:
        mass = 0
        visId = bullet_client.createVisualShape(bullet_client.GEOM_MESH, fileName=os.path.dirname(
            os.path.abspath(__file__)) + '/env_meshes/drawer6.obj', meshScale=[1.25] * 3,
                                                flags=bullet_client.GEOM_FORCE_CONCAVE_TRIMESH,
                                                rgbaColor=[1, 1, 0, 0.5])
    else:
        mass = 0.1
        visId = -1
    drawer_defaults = {"pos": [-0.203+0.265, -0.00, -0.04], "ori": bullet_client.getQuaternionFromEuler([np.pi/2,0,0])}
    drawer = bullet_client.createMultiBody(mass, wallid, visId, drawer_defaults['pos'], baseOrientation=drawer_defaults['ori'])

    if ghostly:
        collisionFilterGroup = 0
        collisionFilterMask = 0
        bullet_client.setCollisionFilterGroupMask(drawer, -1, collisionFilterGroup, collisionFilterMask)

    return {'drawer': drawer, 'defaults':drawer_defaults}


def add_pad(bullet_client, offset=np.array([0, 0, 0]), flags=None, thickness = 1.5, ghostly=False):
    baseOrientation = [0, 0, 0, 1]
    width = 0.17

    colSphereId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                     halfExtents=[width, width, 0.006])
    visualShapeId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                    halfExtents=[width, width, 0.006],
                                                    rgbaColor=[0, 0, 0, 1])
    pad = bullet_client.createMultiBody(0.0, colSphereId, visualShapeId, [0, 0.15, -0.03],
                                                baseOrientation)


    return pad

