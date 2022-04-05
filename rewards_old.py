
rewards = {
"paint": 1.0,
"put_shelf": 1.0,
"put_drawer": 1.0,
"put_side_of_table": 1.0,
"hide": 1.0,
"move": 1.0,
"grasp": 1.0,
}

#TODO: check these:

def check_on_left_side_of_table(pos, state_dict):
    return (-0.3<pos[0]<-0.17) and (0.15-0.17 < pos[1]<0.15+0.17) and (-0.03 < pos[2] < max_size/2)

def check_on_right_side_of_table(pos, state_dict):
    return (0.17<pos[0]<0.3) and (0.15-0.17 < pos[1]<0.15+0.17) and (-0.03 < pos[2] < max_size/2)

def check_on_the_shelf(pos, state_dict):
    return (-0.3<pos[0]<0.3) and (0.15+0.17 < pos[1]<0.15+0.17+0.15) and (-0.03+0.17 < pos[2] < (max_size/2)+0.17)

def check_behind_the_door(pos, state_dict):
    door_pos = state_dict['door_pos'][0]
    return (door_pos-0.2<pos[0]<door_pos+0.2) and (0.15+0.17 < pos[1]<0.15+0.17+0.15) and (-0.03+0.17 < pos[2] < (max_size/2)+0.17)

def check_in_the_drawer(pos, state_dict):
    drawer_pos = state_dict['drawer_pos'][0]
    return (-0.2<pos[0]<0.2) and (drawer_pos-0.1 < pos[1]<drawer_pos+0.2) and (-0.1-0.03+0.17 < pos[2] < -0.1+(max_size/2)+0.17)

position_criteria_funcs={
'on the left side of the table': check_on_left_side_of_table,
'on the right side of the table': check_on_right_side_of_table,
'on the shelf': check_on_the_shelf,
'behind the door': check_behind_the_door,
'in the drawer': check_in_the_drawer
}

from src.envs.color_generation import infer_color

import pickle
types = pickle.load(open("object_types.pkl","rb"))

def paint_reward(type, color, target_color, state_dicts, action):
    matches = 0
    last_state_dict = state_dicts[-1]
    first_state_dict = state_dicts[0]
    for i in range(nb_objs):
        obj_type_index = np.argmax(last_state_dict['obj_{}_type'.format(i)])
        obj_type = types[obj_type_index]
        obj_color_rgb_first = first_state_dict['obj_{}_color'.format(i)]
        obj_color_first = infer_color(obj_color_rgb_first)
        obj_color_rgb_last = last_state_dict['obj_{}_color'.format(i)]
        obj_color_last = infer_color(obj_color_rgb_last)
        if obj_type == type and obj_color_first == color and obj_color_last == target_color:
            matches += 1

    success = matches > 0
    if success:
        return rewards["paint"]
    else:
        return 0

def put_reward(type, color, target_location, state_dicts, action):
    matches = 0
    last_state_dict = state_dicts[-1]
    first_state_dict = state_dicts[0]
    for i in range(nb_objs):
        obj_type_index = np.argmax(state_dict['obj_{}_type'.format(i)])
        obj_type = types[obj_type_index]
        obj_color_rgb_first = state_dict['obj_{}_color'.format(i)]
        obj_color_first = infer_color(obj_color_rgb_first)
        obj_pos_last = state_dict_last['obj_{}_pos'.format(i)]
        location_match_func = position_criteria_funcs[target_location]
        location_match = location_match_func(obj_pos_last, last_state_dict)
        if obj_type == type and obj_color_first == color and location_match:
            matches += 1

    success = matches > 0
    if success:
        return rewards["put"]
    else:
        return 0

def move_reward(type, color, state_dicts, action):
    matches = 0
    last_state_dict = state_dicts[-1]
    first_state_dict = state_dicts[0]
    for i in range(nb_objs):
        obj_type_index = np.argmax(state_dict['obj_{}_type'.format(i)])
        obj_type = types[obj_type_index]
        obj_color_rgb_first = state_dict['obj_{}_color'.format(i)]
        obj_color_first = infer_color(obj_color_rgb_first)
        obj_pos_last = state_dict_last['obj_{}_pos'.format(i)]
        location_match_func = position_criteria_funcs[target_location]
        location_match = location_match_func(obj_pos_last, last_state_dict)
        if obj_type == type and obj_color_first == color and location_match:
            matches += 1

    success = matches > 0
    if success:
        return rewards["put"]
    else:
        return 0
