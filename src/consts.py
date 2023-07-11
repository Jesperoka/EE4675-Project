from os import getcwd
from os.path import isdir, join
from glob import glob
from ast import literal_eval

PROCESSED_DATA_FILENAME = "stacked_spectrums.npz"
PROCESSED_DATA_DIRECTORY = join(getcwd(), "..", "processed_data")
PROCESSED_DATA_PATH = join(PROCESSED_DATA_DIRECTORY, PROCESSED_DATA_FILENAME)

SAMPLE_TIMEBINS = int(512) 

MODEL_DIRECTORY = join(getcwd(), "..", "models")
YAMNET_TFHUB_LINK = "https://tfhub.dev/google/yamnet/1"
YAMNET_MODEL_FOLDER = "yamnet"
YAMNET_MODEL_PATH = join(MODEL_DIRECTORY, YAMNET_MODEL_FOLDER)
YAMNET_EXISTS = isdir(YAMNET_MODEL_PATH)

SEED = 96024

INPUT_SHAPE_FILENAME = "INPUT_SHAPE_"
match = glob(f"*{INPUT_SHAPE_FILENAME}*")
INPUT_SHAPE = literal_eval(match[0][len(INPUT_SHAPE_FILENAME):]) if match else None 
