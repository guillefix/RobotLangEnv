
import os

color_list = ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']

if "ROOT_FOLDER" not in os.environ:
    root_folder="/home/guillefix/code/inria/RobotLangEnv/"
else:
    root_folder = os.environ["ROOT_FOLDER"]

if "PRETRAINED_FOLDER" not in os.environ:
    pretrained_folder="/home/guillefix/code/inria/pretrained/"
else:
    pretrained_folder = os.environ["PRETRAINED_FOLDER"]
if "DATA_FOLDER" not in os.environ:
    data_folder="/home/guillefix/code/inria/UR5/"
else:
    data_folder = os.environ["DATA_FOLDER"]
if "PROCESSED_DATA_FOLDER" not in os.environ:
    processed_data_folder="/home/guillefix/code/inria/UR5_processed/"
else:
    processed_data_folder=os.environ["PROCESSED_DATA_FOLDER"]
if "ROOT_DIR_MODEL" not in os.environ:
    root_dir_model = "/home/guillefix/code/multimodal-transflower/"
else:
    root_dir_model = os.environ["ROOT_DIR_MODEL"]
if "ROOT_DIR_TT_MODEL" not in os.environ:
    root_dir_tt_model = "/home/guillefix/code/trajectory-transformer/"
else:
    root_dir_tt_model = os.environ["ROOT_DIR_TT_MODEL"]
if "ROOT_GENERATED_DATA" not in os.environ:
    root_folder_generated_data = "/home/guillefix/code/inria/"
else:
    root_folder_generated_data = os.environ["ROOT_GENERATED_DATA"]

if "PROCESSED_GENERATED_DATA_FOLDER" not in os.environ:
    processed_generated_data_folder = "/home/guillefix/code/inria/generated_data_processed/"
else:
    processed_generated_data_folder = os.environ["PROCESSED_GENERATED_DATA_FOLDER"]
