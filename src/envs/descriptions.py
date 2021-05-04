# from playground.env_params import get_env_params
from src.envs.env_params import get_env_params


def generate_all_descriptions(env_params):
    """
    Generates all possible descriptions from a set of environment parameters.

    Parameters
    ----------
    env_params: dict
        Dict of environment parameters from get_env_params function.

    Returns
    -------
    training_descriptions: tuple of str
        Tuple of descriptions that belong to the training set (descriptions that do not contain occurrences reserved to the testing set).
    test_descriptions: tuple of str
        Tuple of descriptions that belong to the testing set (that contain occurrences reserved to the testing set).
    all_descriptions: tuple of str
        Tuple of all possible descriptions (training_descriptions + test_descriptions).
    """

    p = env_params.copy()

    # Get the list of admissible attributes and split them by name attributes (type and categories) and other different attributes.
    name_attributes = env_params['name_attributes']
    colors_attributes = env_params['colors_attributes']
    positions_attributes = env_params['positions_attributes']
    drawer_door_attributes = env_params['drawer_door_attributes']
    any_all_attributes = env_params['any_all_attributes']
    rgbb_attributes = env_params['rgbb_attributes']


    all_descriptions = ()
    
    if 'Open' in p['admissible_actions']:
        open_descriptions = []
        for d in drawer_door_attributes:
            open_descriptions.append('Open the {}'.format(d))
        all_descriptions += tuple(open_descriptions)
    

    if 'Close' in p['admissible_actions']:
        close_descriptions = []
        for d in drawer_door_attributes:
            close_descriptions.append('Close the {}'.format(d))
        all_descriptions += tuple(close_descriptions)


    if 'Grasp' in p['admissible_actions']:
        grasp_descriptions = []
        for c in colors_attributes:
            grasp_descriptions.append('Grasp any {} object'.format(c))
        for ca in colors_attributes + ('any',):
            for n in name_attributes:
                grasp_descriptions.append('Grasp {} {}'.format(ca, n))
        all_descriptions += tuple(grasp_descriptions)
    
    
    if 'Put' in p['admissible_actions']:
        put_descriptions = []
        for a in any_all_attributes:
            for c in colors_attributes:
                for pos in positions_attributes:
                    put_descriptions.append('Put {} {} object {}'.format(a, c, pos))
        for ca in colors_attributes + any_all_attributes:
            for n in name_attributes:
                for pos in positions_attributes:
                    put_descriptions.append('Put {} {} {}'.format(ca, n, pos))
        all_descriptions += tuple(put_descriptions)


    if 'Hide' in p['admissible_actions']:
        hide_descriptions = []
        for a in any_all_attributes:
            for c in colors_attributes:
                hide_descriptions.append('Hide {} {} object'.format(a, c))
        for ca in colors_attributes + any_all_attributes:
            for n in name_attributes:
                hide_descriptions.append('Hide {} {}'.format(ca, n))
        all_descriptions += tuple(hide_descriptions)


    if 'Turn on' in p['admissible_actions']:
        turn_on_descriptions = []
        for r in rgbb_attributes:
            turn_on_descriptions.append('Turn on the {} light'.format(r))
        all_descriptions += tuple(turn_on_descriptions)


    if 'Turn off' in p['admissible_actions']:
        turn_off_descriptions = []
        for r in rgbb_attributes:
            turn_off_descriptions.append('Turn off the {} light'.format(r))
        all_descriptions += tuple(turn_off_descriptions)


    if 'Make' in p['admissible_actions']:
        make_descriptions = []
        for c in colors_attributes:
            make_descriptions.append('Make the panel {}'.format(c))
        all_descriptions += tuple(make_descriptions)

    if 'Paint' in p['admissible_actions']:
        color_descriptions = []
        for a in any_all_attributes:
            for c1 in colors_attributes:
                for c2 in sorted(tuple(set(colors_attributes) - set(list(c1)))):
                    color_descriptions.append('Paint {} {} object {}'.format(a, c1, c2))
        for c1 in colors_attributes:
            for n in name_attributes:
                for c2 in sorted(tuple(set(colors_attributes) - set([c1]))):
                    color_descriptions.append('Paint {} {} {}'.format(c1, n, c2))
        for a in any_all_attributes:
            for n in name_attributes:
                for c2 in colors_attributes:
                    color_descriptions.append('Paint {} {} {}'.format(a, n, c2))
        all_descriptions += tuple(color_descriptions)




    train_descriptions = []
    test_descriptions = []
    for descr in all_descriptions:
        to_remove = False
        for w in p['words_test_set_def']: # words_test_set_def is the set of occurrences that is reserved to the testing set.
            if w in descr:
                to_remove = True
                break
        if not to_remove:
            train_descriptions.append(descr)
        else:
            test_descriptions.append(descr)
    
    train_descriptions = tuple(sorted(train_descriptions))
    test_descriptions = tuple(sorted(test_descriptions))

    return train_descriptions, test_descriptions, all_descriptions

if __name__ == '__main__':
    env_params = get_env_params()
    train_descriptions, test_descriptions, all_descriptions = generate_all_descriptions(env_params)
    print("train_descriptions", train_descriptions)
    print("test_descriptions", test_descriptions)
    print("all_descriptions", all_descriptions)
    print("Nb of train descriptions: ", len(train_descriptions))
    print("Nb of test descriptions: ", len(test_descriptions))
    print("Nb of all descriptions: ", len(all_descriptions))

