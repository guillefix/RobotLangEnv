from src.envs.env_params import get_env_params
from src.envs.descriptions import generate_all_descriptions

train_descriptions, test_descriptions, all_descriptions = generate_all_descriptions(get_env_params())


def get_throw_descriptions(get_nb_floor_objects, initial_state, current_state):
    """
    Get all 'throw' descriptions from the current state (if any).
    Parameters
    ----------
    get_nb_floor_objects: function
        Function that gets the number of objects thrown on the floor of the current state compared to the initial state.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.

    Returns
    -------
    descr: list of str
        List of 'throw' descriptions satisfied by the current state.
    """
    throw_descriptions = []
    nb_floor_objects = get_nb_floor_objects(initial_state, current_state)
    for nb in nb_floor_objects:
        throw_descriptions.append('Throw {} objects on the floor'.format(nb))
    return throw_descriptions.copy()


def get_open_descriptions(get_open, initial_state, current_state):
    """
    Get all 'open' descriptions from the current state (if any).
    Parameters
    ----------
    get_open: function
        Function that gets the drawer or the door which is opened.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.

    Returns
    -------
    descr: list of str
        List of 'open' descriptions satisfied by the current state.
    """
    open_descriptions = []
    open_thing = get_open(initial_state, current_state)
    for o in open_thing:
        open_descriptions.append('Open the {}'.format(o))
    return open_descriptions.copy()


def get_close_descriptions(get_close, initial_state, current_state):
    """
    Get all 'close' descriptions from the current state (if any).
    Parameters
    ----------
    get_close: function
        Function that gets the drawer or the door which is closed.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.

    Returns
    -------
    descr: list of str
        List of 'close' descriptions satisfied by the current state.
    """
    close_descriptions = []
    close_thing = get_close(initial_state, current_state)
    for c in close_thing:
        close_descriptions.append('Close the {}'.format(c))
    return close_descriptions.copy()

def get_grasp_descriptions(get_grasped_ids, current_state, sort_attributes, obj_attributes):
    """
    Get all 'grasp' descriptions from the current state (if any).

    Parameters
    ----------
    get_grasped_ids: function
        Function that extracts the id of objects that are being grasped.
    current_state: nd.array
        Current state of the environment.
    sort_attributes: function
        Function that separates adjective and name attributes.
    obj_attributes: list of list
        List of the list of object attributes for each object.

    Returns
    -------
    descr: list of str
        List of 'grasp' descriptions satisfied by the current state.
    """
    obj_grasped = get_grasped_ids(current_state)
    verb = 'Grasp'
    grasp_descriptions = []
    for i_obj in obj_grasped:
        att = obj_attributes[i_obj]
        adj_att, name_att = sort_attributes(att)
        for adj in adj_att:
            quantifier = 'any'
            for name in name_att:
                grasp_descriptions.append('{} {} {}'.format(verb, adj, name))
            grasp_descriptions.append('{} {} {} object'.format(verb, quantifier, adj))
        for name in name_att:
            grasp_descriptions.append('{} any {}'.format(verb, name))

    return grasp_descriptions.copy()


def get_move_descriptions(get_moved_ids, initial_state, current_state, sort_attributes, obj_attributes):
    """
    Get all 'move' descriptions from the current state (if any).

    Parameters
    ----------
    get_moved_ids: function
        Function that extracts the id of objects that are being moved.
    current_state: nd.array
        Current state of the environment.
    sort_attributes: function
        Function that separates adjective and name attributes.
    obj_attributes: list of list
        List of the list of object attributes for each object.

    Returns
    -------
    descr: list of str
        List of 'move' descriptions satisfied by the current state.
    """
    obj_moved = get_moved_ids(initial_state, current_state)
    verb = 'Move'
    move_descriptions = []
    for i_obj in obj_moved:
        att = obj_attributes[i_obj]
        adj_att, name_att = sort_attributes(att)
        for adj in adj_att:
            quantifier = 'any'
            for name in name_att:
                move_descriptions.append('{} {} {}'.format(verb, adj, name))
            move_descriptions.append('{} {} {} object'.format(verb, quantifier, adj))
        for name in name_att:
            move_descriptions.append('{} any {}'.format(verb, name))

    return move_descriptions.copy()


def get_put_descriptions(get_put_ids_pos, initial_state, current_state, sort_attributes, obj_attributes):
    """
    Get all 'put' descriptions from the current state (if any).

    Parameters
    ----------
    get_put_ids_pos: function
        Function that extracts the id and position of objects that are being put.
    current_state: nd.array
        Current state of the environment.
    sort_attributes: function
        Function that separates adjective and name attributes.
    obj_attributes: list of list
        List of the list of object attributes for each object.

    Returns
    -------
    descr: list of str
        List of 'put' descriptions satisfied by the current state.
    """
    obj_put, obj_pos = get_put_ids_pos(initial_state, current_state)
    verb = 'Put'
    put_descriptions = []
    for i_obj in obj_put:
        att = obj_attributes[i_obj]
        adj_att, name_att = sort_attributes(att)
        for adj in adj_att:
            quantifier = 'any'
            for name in name_att:
                put_descriptions.append('{} {} {} {}'.format(verb, adj, name, obj_pos))
            put_descriptions.append('{} {} {} object {}'.format(verb, quantifier, adj, obj_pos))
        for name in name_att:
            put_descriptions.append('{} any {} {}'.format(verb, name, obj_pos))

    return put_descriptions.copy()


def sample_descriptions_from_state(state, params):
    """
    This function samples all description of the current state
    Parameters
    ----------
    state: nd.array
        Current environment state.
    params: dict
        Dict of env parameters.

    Returns
    -------
     descr: list of str
        List of descriptions satisfied by the current state.
    """
    get_grasped_ids = params['extract_functions']['get_interactions']['get_grasped']
    get_grown_ids = params['extract_functions']['get_interactions']['get_grown']
    get_supply_contact = params['extract_functions']['get_interactions']['get_supply_contact']
    get_attributes_functions=params['extract_functions']['get_attributes_functions']
    admissible_attributes = params['admissible_attributes']
    admissible_actions = params['admissible_actions']
    get_obj_features = params['extract_functions']['get_obj_features']
    count_objects = params['extract_functions']['count_objects']
    get_agent_position_attributes = params['extract_functions']['get_agent_position_attributes']
    check_if_relative = params['extract_functions']['check_if_relative']
    combine_two = params['extract_functions']['combine_two']


    current_state = state[:len(state) // 2]
    initial_state = current_state - state[len(state) // 2:]
    assert len(current_state) == len(initial_state)

    nb_objs = count_objects(current_state)
    obj_features = [get_obj_features(initial_state, i_obj) for i_obj in range(nb_objs)]

    # extract object attributes
    obj_attributes = []
    for i_obj in range(nb_objs):
        obj_att = []
        for k in admissible_attributes:
            obj_att += get_attributes_functions[k](obj_features, i_obj)
        obj_attributes.append(obj_att)

    def sort_attributes(attributes):
        adj_attributes = []
        name_attributes = []
        for att in attributes:
            if att in tuple(params['categories'].keys()) + params['attributes']['types']:
                name_attributes.append(att)
            else:
                adj_attributes.append(att)
        return adj_attributes, name_attributes

    descriptions = []

    # Add Move descriptions
    if 'Move' in admissible_actions:
        descriptions += get_move_descriptions(get_agent_position_attributes, current_state)

    # Add Grasp descriptions
    if 'Grasp' in admissible_actions:
        descriptions += get_grasp_descriptions(get_grasped_ids, current_state, sort_attributes, obj_attributes, params, check_if_relative, combine_two)

    # Add Grow descriptions
    if 'Grow' in admissible_actions:
        descriptions += get_grow_descriptions(get_grown_ids, initial_state, current_state, params, obj_attributes, sort_attributes, combine_two, check_if_relative)

        descriptions += get_extra_grow_descriptions(get_supply_contact, initial_state, current_state, params, obj_attributes, sort_attributes, combine_two,
                                                         check_if_relative)
    train_descr = []
    test_descr = []
    extra_descr = []
    for descr in descriptions:
        if descr in train_descriptions:
            train_descr.append(descr)
        elif descr in test_descriptions:
            test_descr.append(descr)
        elif descr in extra_descriptions:
            extra_descr.append(descr)
        else:
            print(descr)
            raise ValueError

    return train_descr.copy(), test_descr.copy(), extra_descr.copy()

def get_reward_from_state(state, goal, params):
    """
    Reward function. Whether the state satisfies the goal.
    Parameters
    ----------
    state: nd.array
        Current environment state.
    goal: str
        Description of the goal.
    params: dict
        Environment parameters.

    Returns
    -------
    bool
    """
    get_grasped_ids = params['extract_functions']['get_interactions']['get_grasped']
    get_grown_ids = params['extract_functions']['get_interactions']['get_grown']
    get_attributes_functions = params['extract_functions']['get_attributes_functions']
    admissible_attributes = params['admissible_attributes']
    admissible_actions = params['admissible_actions']
    get_obj_features = params['extract_functions']['get_obj_features']
    count_objects = params['extract_functions']['count_objects']
    get_agent_position_attributes = params['extract_functions']['get_agent_position_attributes']

    check_if_relative = params['extract_functions']['check_if_relative']
    combine_two = params['extract_functions']['combine_two']

    current_state = state[:len(state) // 2]
    initial_state = current_state - state[len(state) // 2:]
    assert len(current_state) == len(initial_state)

    nb_objs = count_objects(current_state)
    obj_features = [get_obj_features(initial_state, i_obj) for i_obj in range(nb_objs)]

    # extract object attributes
    obj_attributes = []
    for i_obj in range(nb_objs):
        obj_att = []
        for k in admissible_attributes:
            obj_att += get_attributes_functions[k](obj_features, i_obj)
        obj_attributes.append(obj_att)

    def sort_attributes(attributes):
        adj_attributes = []
        name_attributes = []
        for att in attributes:
            if att in tuple(params['categories'].keys()) + params['attributes']['types']:
                name_attributes.append(att)
            else:
                adj_attributes.append(att)
        return adj_attributes, name_attributes


    words = goal.split(' ')
    reward = False

    if words[0] == 'Go':
        go_descr = get_move_descriptions(get_agent_position_attributes, current_state)
        if goal in go_descr:
            reward = True

    if words[0] == 'Grasp':
        grasp_descr = get_grasp_descriptions(get_grasped_ids, current_state, sort_attributes, obj_attributes, params, check_if_relative, combine_two)
        if goal in grasp_descr:
            reward = True

    # Add Grow descriptions
    if words[0] == 'Grow':
        grow_descr = get_grow_descriptions(get_grown_ids, initial_state, current_state, params, obj_attributes, sort_attributes, combine_two, check_if_relative)
        if goal in grow_descr:
            reward = True

    return reward



