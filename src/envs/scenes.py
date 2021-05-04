import numpy as np
import os
TEST_OLD = True

color_dict = dict(red=[1, 0, 0], green=[0, 1, 0], blue=[0, 0, 1], magenta=[1, 0, 1], yellow=[1, 1, 0], cyan=[0, 1, 1], black=[0, 0, 0], white=[1, 1, 1])
table_ranges = [(-0.55, 0.55), (0., 0.35)]
def nothing_scene(bullet_client, offset, flags):

    return []
def default_scene(bullet_client, offset, flags, env_range_low, env_range_high):

    
    # bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], -0.1 + offset[1], -0.6 + offset[2]],
    #                             [-0.5, -0.5, -0.5, 0.5], flags=flags)
    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent,plane_extent, 0.0001])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, plane_extent, 0.0001],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, 0, -0.07])

    return []

def tray_box(bullet_client, offset, flags, env_range_low, env_range_high):
    bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], 0.0 + offset[1], -0.1 + offset[2]],
                                [0,0,0,1], flags=flags)


def push_scene(bullet_client, offset, flags, env_range_low, env_range_high):

    default_scene(bullet_client, offset, flags, env_range_low, env_range_high)
    tray_box(bullet_client, offset, flags, env_range_low, env_range_high)

    legos = []
    side = 0.025
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side],
                                                 rgbaColor=[1, 1, 1, 1])
    block = bullet_client.createMultiBody(0.1, colcubeId, visplaneId, [0, -0.06, -0.06])

    legos.append(block)
        #bullet_client.loadURDF(os.path.dirname(os.path.abspath(__file__)) + "/env_meshes/lego/lego.urdf", np.array([0.1, 0.3, -0.5]) + offset, flags=flags))
    
    return legos


def complex_scene(bullet_client, offset, flags, env_range_low, env_range_high, num_objects):
    #default_scene(bullet_client, offset, flags, env_range_low, env_range_high)
    #tray_box(bullet_client, offset, flags, env_range_low, env_range_high)
    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent, plane_extent, 0.0001])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, plane_extent, 0.0001],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, 0, -0.27])



        
    door = add_door(bullet_client)
    drawer = add_drawer(bullet_client)
    pad = add_pad(bullet_client)#, thickness = thickness) 1.5
    button_red, toggleSphere_red = add_button(bullet_client, position=(-0.48, 0.45), color=(1, 0, 0))
    button_green, toggleSphere_green = add_button(bullet_client, position=(-0.38, 0.45), color=(0, 1, 0))
    button_blue, toggleSphere_blue = add_button(bullet_client, position=(-0.28, 0.45), color=(0, 0, 1))

    # button_red, toggleSphere_red = add_button_red(bullet_client)
    # button_green, toggleSphere_green = add_button_green(bullet_client)
    # button_blue, toggleSphere_blue = add_button_blue(bullet_client)
    # button_black, toggleSphere_black = add_button_black(bullet_client)
    add_static(bullet_client)

    # make objects
    legos = sample_blocks(bullet_client, num_objects)
    # side = 0.025
    # positions = sample_obj_position(num_objects)
    # for b, p in enumerate(positions):
    # # for b in range(0, num_objects):
    #     colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[side * 2, side, side])
    #     visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side * 2, side, side],
    #                                                  rgbaColor=[1, 0, 0, 1])
    #
    #     visplaneId2 = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side * 2, side, side],
    #                                                   rgbaColor=[0, 1, 0, 1])
    #
    #     visplaneId3 = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side * 2, side, side],
    #                                                   rgbaColor=[0, 0, 1, 1])
    #     viz_ids = [visplaneId, visplaneId2, visplaneId3]
    #
    #     legoUID = bullet_client.createMultiBody(0.3, colcubeId, viz_ids[b], p)
    #     bullet_client.changeDynamics(legoUID,
    #                              -1,
    #                             #  spinningFriction=1,
    #                             #  rollingFriction=1,
    #                              lateralFriction=1.5)
    #     legos.append(legoUID)
        
    # return legos, drawer, [door,button_red, button_green, button_blue, button_black, dial], {button_red: ('button_red', toggleSphere_red), button_green: ('button_green', toggleSphere_green), button_blue: ('button_blue', toggleSphere_blue), button_black: ('button_black', toggleSphere_black), dial: ('dial', toggleGrill)} # return the toggle sphere with it's joint index

    return door, drawer, pad, legos, [door,button_red, button_green, button_blue], {button_red: ('button_red', toggleSphere_red), button_green: ('button_green',
                                                                                                                                                 toggleSphere_green),
                                                                         button_blue: ('button_blue', toggleSphere_blue)} # return the toggle sphere with it's joint index

def sample_blocks(bullet_client, num_objects):
    positions = sample_obj_position(num_objects)
    legos = []
    for i in range(num_objects):
        cube_or_block = np.random.choice(['cube', 'block'])
        color = np.random.choice(sorted(color_dict.keys()))
        if cube_or_block == 'block':
            sizes = np.random.uniform(low=0.02, high=0.03, size=2)
            sizes = [2 * sizes[0], sizes[0], sizes[1]]
        else:
            size = np.random.uniform(low=0.02, high=0.03)
            sizes = [size] * 3

        rgb = color_dict[color].copy()
        rgb = perturb_color(rgb)
        for j in range(len(rgb)):
            if rgb[j] == 1:
                rgb[j] -= np.abs(np.random.randn()) * 0.1
            else:
                rgb[j] += np.abs(np.random.randn()) * 0.1
        print(rgb)
        colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=sizes)
        visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=sizes, rgbaColor=rgb + [1])
        positions[i][-1] = sizes[2] / 2
        legoUID = bullet_client.createMultiBody(0.3, colcubeId, visplaneId, positions[i])
        bullet_client.changeDynamics(legoUID,
                                     -1,
                                    #  spinningFriction=1,
                                    #  rollingFriction=1,
                                     lateralFriction=1.5)
        legos.append(legoUID)
    return legos

def perturb_color(rgb):
    rgb = rgb.copy()
    for j in range(len(rgb)):
        if rgb[j] == 1:
            rgb[j] -= np.abs(np.random.randn()) * 0.1
        else:
            rgb[j] += np.abs(np.random.randn()) * 0.1
    return rgb.copy()


def sample_obj_position(nb_objects=3):
    positions = []
    while len(positions) < nb_objects:
        pos = np.array(list(np.random.uniform(low=np.array(table_ranges)[:, 0], high=np.array(table_ranges)[:, 1])) + [0.003])
        j = len(positions)
        while j > 0:
            if np.linalg.norm(positions - pos) < 0.2:
                break
            j -= 1
        positions.append(pos)
    return positions.copy()



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
                                                meshScale = [0.0015] * 3,  flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH, rgbaColor=[0,0,1,0.5])
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
        # bottom = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.265, 0.25, -0.13])
        bottom = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [0, 0.25, -0.13])
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

def dial_to_0_1_range(data):
    return (data % 2*np.pi ) / (2.2*np.pi)

def add_pad(bullet_client, offset=np.array([0, 0, 0]), flags=None, thickness = 1.5, ghostly=False):
    baseOrientation = [0, 0, 0, 1]
    width = 0.17

    colSphereId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                     halfExtents=[width, width, 0.01])
    visualShapeId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                    halfExtents=[width, width, 0.01],
                                                    rgbaColor=[0, 0, 0, 1])
    pad = bullet_client.createMultiBody(0.0, colSphereId, visualShapeId, [0, 0.15, -0.03],
                                                baseOrientation)


    return pad

def add_hinge(bullet_client, offset, flags):
    sphereRadius = 0.05
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                      halfExtents=[sphereRadius, sphereRadius, sphereRadius])

    mass = 0
    visualShapeId = -1

    link_Masses = [1]
    linkCollisionShapeIndices = [colBoxId]
    linkVisualShapeIndices = [-1]
    linkPositions = [[0.0,0.0, -0.5]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([0,np.pi/2,0])]
    linkInertialFramePositions = [[0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    # jointTypes = [bullet_client.JOINT_REVOLUTE]
    jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([0, 0, 0])+offset
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
    return sphereUid
