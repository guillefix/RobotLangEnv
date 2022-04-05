import os
import numpy as np

from src.envs.color_generation import *

def get_env_params(max_nb_objects=3,
                   admissible_actions=('Open', 'Close', 'Grasp', 'Put', 'Hide', 'Turn on', 'Turn off', 'Make', 'Paint', 'Move', 'Throw'),
                   admissible_attributes=('colors', 'categories', 'types'),
                   min_max_sizes=(0.1, 0.15),
                   agent_size=0.05,
                   epsilon_initial_pos=0.3,
                   screen_size=800,
                   next_to_epsilon=0.3,
                   table_ranges = ((-0.42, 0.42), (0.05, 0.25)),
                   attribute_combinations=False,
                   obj_size_update=0.04,
                   render_mode=True
                   ):
    """
    Builds the set of environment parameters, and the set of function to extract information from the state.

    Parameters
    ----------
    max_nb_objects: int
         Maximum number of objects in the scene (effective number if it's not random).
    admissible_actions: tuple of str
        which types of actions are admissible
    admissible_attributes: tuple of str
        All admissible attributes, should be included in ('colors', 'categories', 'types', 'relative_sizes', 'shades', 'relative_shades', 'sizes', 'relative_positions')
    min_max_sizes: tuple of tuples
        Min and max sizes for the small and big objects respectively.
    agent_size: float
        Size of the agent.
    epsilon_initial_pos: float
        Range of initial position around origin.
    screen_size: int
        Screen size in pixels.
    next_to_epsilon: float
        Define which area corresponds to 'next to'.
    attribute_combinations: Bool
        Whether attributes should include combinations of two attributes.
    obj_size_update: float
        By how much should be updated the size of objects when the agent grows them.
    render_mode: Bool
        Whether to render the environment.

    Returns
    -------
    params: dict
    """

    # list objects and categories
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
    # List types
    types = ()
    for k_c in categories.keys():
        types += categories[k_c]
    # types = tuple(set(types)) # filters doubles, when some categories include others.
    import pickle
    # import os
    # print(os.getcwd())
    if "ROOT_FOLDER" not in os.environ:
        root_folder="/home/guillefix/code/inria/captionRLenv/"
    else:
        root_folder = os.environ["ROOT_FOLDER"]
    types = pickle.load(open(root_folder+"object_types.pkl","rb"))
    nb_types = len(types)

    # List attributes + others
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

    # Get the list of admissible attributes
    name_attributes = ()
    colors_attributes = ()
    positions_attributes = ()
    drawer_door_attributes = ()
    any_all_attributes = ()
    rgbb_attributes = ()
    for att_type in attributes.keys():
        if att_type in ('types', 'categories'):
            name_attributes += attributes[att_type]
        elif att_type == 'colors':
            colors_attributes += attributes[att_type]
        elif att_type == 'positions':
            positions_attributes += attributes[att_type]
        elif att_type == 'drawer_door':
            drawer_door_attributes += attributes[att_type]
        elif att_type == 'any_all':
            any_all_attributes += attributes[att_type]
        elif att_type == 'rgbb':
            rgbb_attributes += attributes[att_type]



    # This defines the list of occurrences that should belong to the test set. All descriptions that contain them belong to the test set.
    words_test_set_def = ('red bear', 'green donut', 'blue bowl', 'white cube', 'black car') + \
                         ('bird',) + \
                         ('apple on the right side of the table',) + \
                         tuple('Grasp {} food'.format(c) for c in colors + ('any',)) + \
                         tuple('Move {} bottle'.format(c) for c in colors + ('any',)) + \
                         tuple('Hide {} {}'.format(c, v) for c in colors + any_all for v in vehicles_model) + \
                         tuple('Put yellow {} in the drawer'.format(t) for t in name_attributes) + \
                         tuple('Paint black {} magenta'.format(t) for t in name_attributes)



    # get indices of attributes in object feature vector
    dim_body_features = 3
    agent_position_inds = np.arange(2)
    dim_obj_features = nb_types + 7
    type_inds = np.arange(0, nb_types)
    position_inds = np.arange(nb_types, nb_types + 2)
    size_inds = np.array(nb_types + 2)
    color_inds = np.arange(nb_types + 3, nb_types + 6)
    grasped_inds = np.array([nb_types + 6])

    params = dict(nb_types=nb_types,
                  max_nb_objects=max_nb_objects,
                  admissible_actions=admissible_actions,
                  admissible_attributes=admissible_attributes,
                  dim_body_features=dim_body_features,
                  table_ranges=table_ranges,
                  agent_position_inds=agent_position_inds,
                  grasped_inds=grasped_inds,
                  attributes=attributes,
                  categories=categories,
                  types=types,
                  name_attributes=name_attributes,
                  colors_attributes = colors_attributes,
                  positions_attributes = positions_attributes,
                  drawer_door_attributes = drawer_door_attributes,
                  any_all_attributes = any_all_attributes,
                  rgbb_attributes = rgbb_attributes,
                  words_test_set_def=words_test_set_def,
                  dim_obj_features=dim_obj_features,  # one-hot of things, 2D position, size, rgb code, grasped Boolean
                  color_inds=color_inds,
                  size_inds=size_inds,
                  position_inds=position_inds,
                  type_inds=type_inds,
                  min_max_sizes=min_max_sizes,
                  agent_size=agent_size,
                  epsilon_initial_pos=epsilon_initial_pos,
                  screen_size=screen_size,
                  ratio_size=int(screen_size / 2.4),
                  next_to_epsilon=next_to_epsilon,
                  attribute_combinations=attribute_combinations,
                  obj_size_update=obj_size_update,
                  render_mode=render_mode
                  )

    # # # # # # # # # # # # # # #
    # Define extraction functions
    # # # # # # # # # # # # # # #

    # Get the number of objects thrown on the floor
    def get_nb_floor_objects(initial_state, current_state):
        nb = 0
        for i in range(3):
            if (initial_state[10 + i * 35] > -0.17) and (current_state[10 + i * 35] < -0.2):
                nb = nb + 1
        return nb

    # Extract interactions with objects
    def get_open(initial_state, current_state):
        open = []
        if (-0.05 < initial_state[113] < 0.05) and (current_state[113] < -0.22):
            open.append('door')
        if (initial_state[114] - current_state[114]) > 0.00001:
            open.append('drawer')
        return open

    def get_close(initial_state, current_state):
        close = []
        if (-0.05 < initial_state[113] < 0.05) and (current_state[113] > 0.22):
            close.append('door')
        if current_state[114] > 0.06:
            close.append('drawer')
        return close

    def get_grasped_ids(current_state):
        obj_grasped = []
        if current_state[7] > 0.03:
            for i in range(3):
                if current_state[10 + i * 35] > 0.08:
                    obj_grasped.append(i)
        return obj_grasped

    def get_moved_ids(initial_state, current_state):
        obj_moved = []
        for i in range(3):
            if (abs(current_state[8 + i * 35] - initial_state[8 + i * 35]) > 0.0001) or (abs(current_state[9 + i * 35] - initial_state[9 + i * 35]) > 0.0001):
                obj_moved.append(i)
        return obj_moved

    def get_put_ids_pos(initial_state, current_state):
        obj_put = []
        obj_pos = {}
        door_pos = current_state[113]
        drawer_pos = current_state[114]
        for i in range(3):
            pos = current_state[8 + i * 35:11 + i * 35]
            size = current_state[i*35+8+32:i*35+8+35]
            vertical_size = np.max(size)
            init_pos = initial_state[8 + i * 35:11 + i * 35]
            if (abs(pos[0] - init_pos[0]) > 0.0001) or (abs(pos[1] - init_pos[1]) > 0.0001):
                obj_put.append(i)
                if (-0.6 <= pos[0] < 0) and (0 <= pos[1] <= 0.32) and (-0.04 <= pos[2] <= vertical_size/2):
                    obj_pos[i] = 'on the left side of the table'
                elif (0.6 >= pos[0] > 0) and (0 <= pos[1] <= 0.32) and (-0.04 <= pos[2] <= vertical_size/2):
                    obj_pos[i] = 'on the right side of the table'
                elif (-0.6 <= pos[0] <= 0.6) and (0.32 <= pos[1] < 0.6) and ( 0.22 < pos[2] < 0.32):
                    obj_pos[i] = 'on the shelf'
                elif (door_pos-0.17 < pos[0] < door_pos+0.17) and (0.32 <= pos[1] < 0.6) and (-0.04 <= pos[2] <= 0.22):
                    obj_pos[i] = 'behind the door'
                elif (-0.17 <= pos[0] <= 0.17) and (drawer_pos - 0.1 < pos[1] < drawer_pos + 0.14) and (-0.17 <= pos[2] < -0.04):
                    obj_pos[i] = 'in the drawer'
                else:
                    obj_put.remove(i)
        return obj_put, obj_pos

    def get_hidden_ids(current_state):
        obj_hidden = []
        for i in range(3):
            if (current_state[9 + i * 35] > 0.32) and (-0.04 <= current_state[10 + i * 35] <= 0.24) and (current_state[113] > 0.22):
                obj_hidden.append(i)
            elif (-0.17 <= current_state[10 + i * 35] < -0.04) and (current_state[114] > 0.06):
                obj_hidden.append(i)
        return obj_hidden

    def get_turn_on(initial_state, current_state):
        light_on = []
        if (initial_state[120] == 0) and (current_state[120] == 1):
            light_on.append('red')
        if (initial_state[122] == 0) and (current_state[122] == 1):
            light_on.append('green')
        if (initial_state[124] == 0) and (current_state[124] == 1):
            light_on.append('blue')
        return light_on

    def get_turn_off(initial_state, current_state):
        light_off = []
        if (initial_state[120] == 1) and (current_state[120] == 0):
            light_off.append('red')
        if (initial_state[122] == 1) and (current_state[122] == 0):
            light_off.append('green')
        if (initial_state[124] == 1) and (current_state[124] == 0):
            light_off.append('blue')
        return light_off

    def get_make(current_state):
        if (current_state[115] == 0) and (current_state[116] == 0) and (current_state[117] == 0):
            panel_color = 'black'
        elif (current_state[115] == 1) and (current_state[116] == 0) and (current_state[117] == 0):
            panel_color = 'red'
        elif (current_state[115] == 0) and (current_state[116] == 1) and (current_state[117] == 0):
            panel_color = 'green'
        elif (current_state[115] == 0) and (current_state[116] == 0) and (current_state[117] == 1):
            panel_color = 'blue'
        elif (current_state[115] == 1) and (current_state[116] == 1) and (current_state[117] == 0):
            panel_color = 'yellow'
        elif (current_state[115] == 1) and (current_state[116] == 0) and (current_state[117] == 1):
            panel_color = 'magenta'
        elif (current_state[115] == 0) and (current_state[116] == 1) and (current_state[117] == 1):
            panel_color = 'cyan'
        elif (current_state[115] == 1) and (current_state[116] == 1) and (current_state[117] == 1):
            panel_color = 'white'
        return panel_color

    def get_paint(initial_state, current_state, obj_stuff):
        obj_paint = []
        obj_name = {}
        color_init = {}
        color_fina = {}
        for i in range(3):
            if (current_state[37 + i * 35] != initial_state[37 + i * 35]) or (current_state[38 + i * 35] != initial_state[38 + i * 35]) or (current_state[39 + i * 35] != initial_state[39 + i * 35]):
                obj_paint.append(i)
            name_attributes = []
            name_attributes.append(obj_stuff[0][i]['type'])     # append painted object type to name_attributes
            for key, value in params['categories'].items():
                if obj_stuff[0][i]['type'] in value:
                    name_attributes.append(key)         # append painted object category to name_attributes
            obj_name[i] = name_attributes
            color_init[i] = obj_stuff[0][i]['color']
            rgb_final = np.array([current_state[37 + i * 35], current_state[38 + i * 35], current_state[39 + i * 35]])
            color_fina[i] = infer_color(rgb_final)
        return obj_paint, obj_name, color_init, color_fina


    get_interactions = dict(get_open=get_open,
                            get_close=get_close,
                            get_grasped=get_grasped_ids,
                            get_moved=get_moved_ids,
                            get_put=get_put_ids_pos,
                            get_hidden=get_hidden_ids,
                            get_turn_on=get_turn_on,
                            get_turn_off=get_turn_off,
                            get_make=get_make,
                            get_paint=get_paint)


    params['extract_functions'] = dict(get_nb_floor_objects=get_nb_floor_objects,
                                       get_interactions=get_interactions)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    return params
