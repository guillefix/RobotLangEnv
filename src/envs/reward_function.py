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
    throw_descriptions.append('Throw {} objects on the floor'.format(nb_floor_objects))
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

def get_grasp_descriptions(get_grasped_ids, current_state, obj_stuff, sort_attributes, obj_attributes):
    """
    Get all 'grasp' descriptions from the current state (if any).

    Parameters
    ----------
    get_grasped_ids: function
        Function that extracts the id of objects that are being grasped.
    current_state: nd.array
        Current state of the environment.
    obj_stuff: list of objects and their sizes
        List of initial objects {type, color, category} and their sizes.
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


def get_move_descriptions(get_moved_ids, initial_state, current_state, obj_stuff, sort_attributes, obj_attributes):
    """
    Get all 'move' descriptions from the current state (if any).

    Parameters
    ----------
    get_moved_ids: function
        Function that extracts the id of objects that are being moved.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.
    obj_stuff: list of objects and their sizes
        List of initial objects {type, color, category} and their sizes.
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


def get_put_descriptions(get_put_ids_pos, initial_state, current_state, obj_stuff, sort_attributes, obj_attributes):
    """
    Get all 'put' descriptions from the current state (if any).

    Parameters
    ----------
    get_put_ids_pos: function
        Function that extracts the id and position of objects that are being put.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.
    obj_stuff: list of objects and their sizes
        List of initial objects {type, color, category} and their sizes.
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
                put_descriptions.append('{} {} {} {}'.format(verb, adj, name, obj_pos[i_obj]))
            put_descriptions.append('{} {} {} object {}'.format(verb, quantifier, adj, obj_pos[i_obj]))
        for name in name_att:
            put_descriptions.append('{} any {} {}'.format(verb, name, obj_pos[i_obj]))
    for a_obj in obj_put:
        all_quant = True
        for b_obj in list(set([0,1,2]) - set([a_obj])):
            if sort_attributes(obj_attributes[b_obj])[0] == sort_attributes(obj_attributes[a_obj])[0]:
                if b_obj in obj_put:
                    if obj_pos[b_obj] != obj_pos[a_obj]:
                        all_quant = False
                else:
                    all_quant = False
        if all_quant == True:
            for col in sort_attributes(obj_attributes[a_obj])[0]:
                des = '{} all {} object {}'.format(verb, col, obj_pos[a_obj])
                if des not in put_descriptions:
                    put_descriptions.append(des)
    for c_obj in obj_put:
        all_quant2 = True
        for d_obj in list(set([0,1,2]) - set([c_obj])):
            if sort_attributes(obj_attributes[d_obj])[1][0] == sort_attributes(obj_attributes[c_obj])[1][0]:
                if d_obj in obj_put:
                    if obj_pos[d_obj] != obj_pos[c_obj]:
                        all_quant2 = False
                else:
                    all_quant2 = False
        if all_quant2 == True:
            type = sort_attributes(obj_attributes[c_obj])[1][0]
            des = '{} all {} {}'.format(verb, type, obj_pos[c_obj])
            if des not in put_descriptions:
                put_descriptions.append(des)
    for e_obj in obj_put:
        all_quant3 = True
        for f_obj in list(set([0,1,2]) - set([e_obj])):
            if sort_attributes(obj_attributes[f_obj])[1][1] == sort_attributes(obj_attributes[e_obj])[1][1]:
                if f_obj in obj_put:
                    if obj_pos[f_obj] != obj_pos[e_obj]:
                        all_quant3 = False
                else:
                    all_quant3 = False
        if all_quant3 == True:
            cat = sort_attributes(obj_attributes[e_obj])[1][1]
            des = '{} all {} {}'.format(verb, cat, obj_pos[e_obj])
            if des not in put_descriptions:
                put_descriptions.append(des)


    return put_descriptions.copy()


def get_hide_descriptions(get_hidden_ids, initial_state, current_state, obj_stuff, sort_attributes, obj_attributes):
    """
    Get all 'hide' descriptions from the current state (if any).

    Parameters
    ----------
    get_hidden_ids: function
        Function that extracts the id of objects that are being hidden.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.
    obj_stuff: list of objects and their sizes
        List of initial objects {type, color, category} and their sizes.
    sort_attributes: function
        Function that separates adjective and name attributes.
    obj_attributes: list of list
        List of the list of object attributes for each object.

    Returns
    -------
    descr: list of str
        List of 'hide' descriptions satisfied by the current state.
    """
    obj_hidden = get_hidden_ids(current_state)
    verb = 'Hide'
    hide_descriptions = []
    for i_obj in obj_hidden:
        att = obj_attributes[i_obj]
        adj_att, name_att = sort_attributes(att)
        for adj in adj_att:
            quantifier = 'any'
            for name in name_att:
                hide_descriptions.append('{} {} {}'.format(verb, adj, name))
            hide_descriptions.append('{} {} {} object'.format(verb, quantifier, adj))
        for name in name_att:
            hide_descriptions.append('{} any {}'.format(verb, name))
    for a_obj in obj_hidden:
        all_quant = True
        for b_obj in list(set([0,1,2]) - set([a_obj])):
            if sort_attributes(obj_attributes[b_obj])[0] == sort_attributes(obj_attributes[a_obj])[0]:
                if b_obj not in obj_hidden:
                    all_quant = False
        if all_quant == True:
            for col in sort_attributes(obj_attributes[a_obj])[0]:
                des = '{} all {} object'.format(verb, col)
                if des not in hide_descriptions:
                    hide_descriptions.append(des)
    for c_obj in obj_hidden:
        all_quant2 = True
        for d_obj in list(set([0,1,2]) - set([c_obj])):
            if sort_attributes(obj_attributes[d_obj])[1][0] == sort_attributes(obj_attributes[c_obj])[1][0]:
                if d_obj not in obj_hidden:
                    all_quant2 = False
        if all_quant2 == True:
            type = sort_attributes(obj_attributes[c_obj])[1][0]
            des = '{} all {}'.format(verb, type)
            if des not in hide_descriptions:
                hide_descriptions.append(des)
    for e_obj in obj_hidden:
        all_quant3 = True
        for f_obj in list(set([0,1,2]) - set([e_obj])):
            if sort_attributes(obj_attributes[f_obj])[1][1] == sort_attributes(obj_attributes[e_obj])[1][1]:
                if f_obj not in obj_hidden:
                    all_quant3 = False
        if all_quant3 == True:
            cat = sort_attributes(obj_attributes[e_obj])[1][1]
            des = '{} all {}'.format(verb, cat)
            if des not in hide_descriptions:
                hide_descriptions.append(des)

    return hide_descriptions.copy()


def get_turn_on_descriptions(get_turn_on, initial_state, current_state):
    """
    Get all 'turn on' descriptions from the current state (if any).
    Parameters
    ----------
    get_turn_on: function
        Function that gets the color of light which is on.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.

    Returns
    -------
    descr: list of str
        List of 'turn on' descriptions satisfied by the current state.
    """
    turn_on_descriptions = []
    light_on = get_turn_on(initial_state, current_state)
    for c in light_on:
        turn_on_descriptions.append('Turn on the {} light'.format(c))
    return turn_on_descriptions.copy()


def get_turn_off_descriptions(get_turn_off, initial_state, current_state):
    """
    Get all 'turn off' descriptions from the current state (if any).
    Parameters
    ----------
    get_turn_off: function
        Function that gets the color of light which is off.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.

    Returns
    -------
    descr: list of str
        List of 'turn off' descriptions satisfied by the current state.
    """
    turn_off_descriptions = []
    light_off = get_turn_off(initial_state, current_state)
    for c in light_off:
        turn_off_descriptions.append('Turn off the {} light'.format(c))
    return turn_off_descriptions.copy()


def get_make_descriptions(get_make, initial_state, current_state):
    """
    Get all 'make' descriptions from the current state (if any).
    Parameters
    ----------
    get_make: function
        Function that gets the current color of panel.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.

    Returns
    -------
    descr: list of str
        List of 'make' descriptions satisfied by the current state.
    """
    make_descriptions = []
    panel_color = get_make(current_state)
    make_descriptions.append('Make the panel {}'.format(panel_color))
    return make_descriptions.copy()


def get_paint_descriptions(get_paint, initial_state, current_state, obj_stuff):
    """
    Get all 'paint' descriptions from the current state (if any).

    Parameters
    ----------
    get_paint: function
        Function that extracts the id of objects that are being painted and its initial color as well as its final color.
    initial_state: nd.array
        Initial state of the environment.
    current_state: nd.array
        Current state of the environment.
    obj_stuff: list of objects and their sizes
        List of initial objects {type, color, category} and their sizes.

    Returns
    -------
    descr: list of str
        List of 'paint' descriptions satisfied by the current state.
    """
    obj_paint, obj_name, color_init, color_fina = get_paint(initial_state, current_state, obj_stuff)
    verb = 'Paint'
    paint_descriptions = []
    for i_obj in obj_paint:
        name_att = obj_name[i_obj]
        color_init_att = [color_init[i_obj]]
        color_fina_att = [color_fina[i_obj]]
        for c1 in color_init_att:
            quantifier = 'any'
            for c2 in color_fina_att:
                for name in name_att:
                    paint_descriptions.append('{} {} {} {}'.format(verb, c1, name, c2))
                paint_descriptions.append('{} {} {} object {}'.format(verb, quantifier, c1, c2))
        for c2 in color_fina_att:
                for name in name_att:
                    paint_descriptions.append('{} any {} {}'.format(verb, name, c2))
    for a_obj in obj_paint:
        all_quant = True
        for b_obj in list(set([0,1,2]) - set([a_obj])):
            if obj_stuff[0][b_obj]['color'] == obj_stuff[0][a_obj]['color']:
                if b_obj not in obj_paint:
                    all_quant = False
        if all_quant == True:
            for c_init in [color_init[a_obj]]:
                for c_fina in [color_fina[a_obj]]:
                    des = '{} all {} object {}'.format(verb, c_init, c_fina)
                    if des not in paint_descriptions:
                        paint_descriptions.append(des)
    for c_obj in obj_paint:
        all_quant2 = True
        for d_obj in list(set([0,1,2]) - set([c_obj])):
            if obj_stuff[0][d_obj]['type'] == obj_stuff[0][c_obj]['type']:
                if color_fina[d_obj] != color_fina[c_obj]:
                    all_quant2 = False
        if all_quant2 == True:
            type = obj_stuff[0][c_obj]['type']
            des = '{} all {} {}'.format(verb, type, color_fina[c_obj])
            if des not in paint_descriptions:
                paint_descriptions.append(des)
    for e_obj in obj_paint:
        all_quant3 = True
        for f_obj in list(set([0,1,2]) - set([e_obj])):
            if obj_name[f_obj][1] == obj_name[e_obj][1]:
                if color_fina[f_obj] != color_fina[e_obj]:
                    all_quant3 = False
        if all_quant3 == True:
            cat = obj_name[e_obj][1]
            des = '{} all {} {}'.format(verb, cat, color_fina[e_obj])
            if des not in paint_descriptions:
                paint_descriptions.append(des)

    return paint_descriptions.copy(), obj_paint, obj_name, color_init


def sample_descriptions_from_state(initial_state, current_state, obj_stuff, params):
    """
    This function samples all descriptions of the current state compared to the initial state
    Parameters
    ----------
    initial_state: nd.array
        Initial environment state.
    current_state: nd.array
        Current environment state.
    obj_stuff: list of objects and their sizes
        List of initial objects {type, color, category} and their sizes.
    params: dict
        Dict of env parameters.

    Returns
    -------
     descr: list of str
        List of descriptions satisfied by the current state compared to the initial state.
    """
    get_nb_floor_objects = params['extract_functions']['get_nb_floor_objects']
    get_open = params['extract_functions']['get_interactions']['get_open']
    get_close = params['extract_functions']['get_interactions']['get_close']
    get_grasped_ids = params['extract_functions']['get_interactions']['get_grasped']
    get_moved_ids = params['extract_functions']['get_interactions']['get_moved']
    get_put_ids_pos = params['extract_functions']['get_interactions']['get_put']
    get_hidden_ids = params['extract_functions']['get_interactions']['get_hidden']
    get_turn_on = params['extract_functions']['get_interactions']['get_turn_on']
    get_turn_off = params['extract_functions']['get_interactions']['get_turn_off']
    get_make = params['extract_functions']['get_interactions']['get_make']
    get_paint = params['extract_functions']['get_interactions']['get_paint']
    admissible_actions = params['admissible_actions']


    assert len(current_state) == len(initial_state)

    obj_features_init = obj_stuff[0]       # initial list of dict for {type, color, category=None}

    # # extract object attributes
    # obj_attributes = []
    # for i_obj in range(nb_objs):
    #     obj_att = []
    #     for k in admissible_attributes:
    #         obj_att += get_attributes_functions[k](obj_features, i_obj)
    #     obj_attributes.append(obj_att)

    def sort_attributes(attributes):
        adj_attributes = []
        name_attributes = []
        for att in attributes:
            if att=='type':
                name_attributes.append(attributes[att])     # append object type to name_attributes
                for key, value in params['categories'].items():
                    if attributes[att] in value:
                        name_attributes.append(key)         # append object category to name_attributes
            elif att=='color':
                adj_attributes.append(attributes[att])          # append current object color to adj_attributes
        return adj_attributes, name_attributes


    descriptions = []

    # Add Throw descriptions
    if 'Throw' in admissible_actions:
        descriptions += get_throw_descriptions(get_nb_floor_objects, initial_state, current_state)

    # Add Open descriptions
    if 'Open' in admissible_actions:
        descriptions += get_open_descriptions(get_open, initial_state, current_state)

    # Add Close descriptions
    if 'Close' in admissible_actions:
        descriptions += get_close_descriptions(get_close, initial_state, current_state)

    # Add Grasp descriptions
    if 'Grasp' in admissible_actions:
        descriptions += get_grasp_descriptions(get_grasped_ids, current_state, obj_stuff, sort_attributes, obj_features_init)

    # Add Move descriptions
    if 'Move' in admissible_actions:
        descriptions += get_move_descriptions(get_moved_ids, initial_state, current_state, obj_stuff, sort_attributes, obj_features_init)

    # Add Put descriptions
    if 'Put' in admissible_actions:
        descriptions += get_put_descriptions(get_put_ids_pos, initial_state, current_state, obj_stuff, sort_attributes, obj_features_init)

    # Add Hide descriptions
    if 'Hide' in admissible_actions:
        descriptions += get_hide_descriptions(get_hidden_ids, initial_state, current_state, obj_stuff, sort_attributes, obj_features_init)

    # Add Turn on descriptions
    if 'Turn on' in admissible_actions:
        descriptions += get_turn_on_descriptions(get_turn_on, initial_state, current_state)

    # Add Turn off descriptions
    if 'Turn off' in admissible_actions:
        descriptions += get_turn_off_descriptions(get_turn_off, initial_state, current_state)

    # Add Make descriptions
    if 'Make' in admissible_actions:
        descriptions += get_make_descriptions(get_make, initial_state, current_state)

    # Add Paint descriptions
    if 'Paint' in admissible_actions:
        paint_descriptions, obj_paint, obj_name, color_init = get_paint_descriptions(get_paint, initial_state, current_state, obj_stuff)
        descriptions += paint_descriptions


    paint_action = False
    for d in descriptions:
        words = d.split(' ')
        if words[0] == 'Paint':
            paint_action = True
            break
    if paint_action == True:
        for i in obj_paint:
            type = obj_name[i][0]
            cat = obj_name[i][1]
            color = color_init[i]
            remove_type = color + ' ' + type
            remove_cat = color + ' ' + cat
            remove_obj = color + ' ' + 'object'
            for des in reversed(descriptions):
                words = des.split(' ')
                if words[0] != 'Paint':
                    if (des.find(remove_type) != -1) or (des.find(remove_cat) != -1) or (des.find(remove_obj) != -1):
                        descriptions.remove(des)

    train_descr = []
    test_descr = []
    for descr in descriptions:
        if descr in train_descriptions:
            train_descr.append(descr)
        elif descr in test_descriptions:
            test_descr.append(descr)

    return train_descr.copy(), test_descr.copy()

def get_reward_from_state(initial_state, current_state, obj_stuff, goal, params):
    """
    Reward function. Whether the state satisfies the goal.
    Parameters
    ----------
    initial_state: nd.array
        Initial environment state.
    current_state: nd.array
        Current environment state.
    obj_stuff: list of objects and their sizes
        List of initial objects {type, color, category} and their sizes.
    goal: str
        Description of the goal.
    params: dict
        Environment parameters.

    Returns
    -------
    reward: bool
    """
    get_nb_floor_objects = params['extract_functions']['get_nb_floor_objects']
    get_open = params['extract_functions']['get_interactions']['get_open']
    get_close = params['extract_functions']['get_interactions']['get_close']
    get_grasped_ids = params['extract_functions']['get_interactions']['get_grasped']
    get_moved_ids = params['extract_functions']['get_interactions']['get_moved']
    get_put_ids_pos = params['extract_functions']['get_interactions']['get_put']
    get_hidden_ids = params['extract_functions']['get_interactions']['get_hidden']
    get_turn_on = params['extract_functions']['get_interactions']['get_turn_on']
    get_turn_off = params['extract_functions']['get_interactions']['get_turn_off']
    get_make = params['extract_functions']['get_interactions']['get_make']
    get_paint = params['extract_functions']['get_interactions']['get_paint']
    admissible_actions = params['admissible_actions']


    assert len(current_state) == len(initial_state)

    obj_features_init = obj_stuff[0]       # initial list of dict for {type, color, category=None}

    # # extract object attributes
    # obj_attributes = []
    # for i_obj in range(nb_objs):
    #     obj_att = []
    #     for k in admissible_attributes:
    #         obj_att += get_attributes_functions[k](obj_features, i_obj)
    #     obj_attributes.append(obj_att)

    def sort_attributes(attributes):
        adj_attributes = []
        name_attributes = []
        for att in attributes:
            if att=='type':
                name_attributes.append(attributes[att])     # append object type to name_attributes
                for key, value in params['categories'].items():
                    if attributes[att] in value:
                        name_attributes.append(key)         # append object category to name_attributes
            elif att=='color':
                adj_attributes.append(attributes[att])          # append current object color to adj_attributes
        return adj_attributes, name_attributes


    words = goal.split(' ')
    reward = False

    if words[0] == 'Throw':
        throw_descr = get_throw_descriptions(get_nb_floor_objects, initial_state, current_state)
        if goal in throw_descr:
            reward = True

    if words[0] == 'Open':
        open_descr = get_open_descriptions(get_open, initial_state, current_state)
        if goal in open_descr:
            reward = True

    if words[0] == 'Close':
        close_descr = get_close_descriptions(get_close, initial_state, current_state)
        if goal in close_descr:
            reward = True

    if words[0] == 'Grasp':
        grasp_descr = get_grasp_descriptions(get_grasped_ids, current_state, obj_stuff, sort_attributes, obj_features_init)
        if goal in grasp_descr:
            reward = True

    if words[0] == 'Move':
        move_descr = get_move_descriptions(get_moved_ids, initial_state, current_state, obj_stuff, sort_attributes, obj_features_init)
        if goal in move_descr:
            reward = True

    if words[0] == 'Put':
        put_descr = get_put_descriptions(get_put_ids_pos, initial_state, current_state, obj_stuff, sort_attributes, obj_features_init)
        if goal in put_descr:
            reward = True

    if words[0] == 'Hide':
        hide_descr = get_hide_descriptions(get_hidden_ids, initial_state, current_state, obj_stuff, sort_attributes, obj_features_init)
        if goal in hide_descr:
            reward = True

    if words[0] == 'Turn':
        if words[1] == 'on':
            turn_on_descr = get_turn_on_descriptions(get_turn_on, initial_state, current_state)
            if goal in turn_on_descr:
                reward = True
        elif words[1] == 'off':
            turn_off_descr = get_turn_off_descriptions(get_turn_off, initial_state, current_state)
            if goal in turn_off_descr:
                reward = True

    if words[0] == 'Make':
        make_descr = get_make_descriptions(get_make, initial_state, current_state)
        if goal in make_descr:
            reward = True

    if words[0] == 'Paint':
        paint_descr, _, _, _ = get_paint_descriptions(get_paint, initial_state, current_state, obj_stuff)
        if goal in paint_descr:
            reward = True


    return reward
