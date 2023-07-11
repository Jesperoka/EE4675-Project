"""Preprocesing of data"""
from os import getcwd, listdir, mkdir
from os.path import isdir, join
from pathlib import Path
from random import randint
from gc import collect

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage
from numba import jit
from numba.typed import List
from tqdm import tqdm
from re import match

import consts
from util import jit_abs, normalize_array

SAMPLE_TIME = 0.0082  # sample time in seconds
MIN_RANGE = 1  # minimum range in meters
MAX_RANGE = 4.8  # maximum range in meters


def main():
    RAW_DATA_DIRECTORY = join(getcwd(), "..", "raw_data")
    PROCESSED_DATA_DIRECTORY = join(getcwd(), "..", "processed_data")

    print("\n\n\nProcessing the raw data in", RAW_DATA_DIRECTORY, ". . . this will take some time . . .\n\n\n")

    preprocess(RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY)

    print("\n\n\nDone! The processed data is in", PROCESSED_DATA_DIRECTORY, "\n\n\n")


def preprocess(input_directory_path, output_directory_path):
    """
    Load, analyze and preprocess all data in the given input directory,
    and output the processed data to the given output directory.
    """

    data_arrays, classes = load_raw_data(input_directory_path, ".mat", loadmat_first_dict_radar_three_data_only)

    data_arrays = average_out_zeros(data_arrays)

    # analyze_data_quality(data_arrays)

    samples_stacked_by_class = split_data_arrays_into_stacked_samples(data_arrays)
    collect()
    #split_data_arrays = np.sort(split_data_arrays, axis=-2)[:, :, -1:-(25+1):-1, :] # WARNING: good or bad?

    # Need to compute iteratively due to OOM errors when allocating array
    # amount = 64
    saved_shape_once = False
    print("\n\n\nEntering main processing loop\n\n\n")
    for some_samples_in_a_category, category in tqdm(zip(samples_stacked_by_class, classes), total=len(classes)):
        category_directory = join(consts.PROCESSED_DATA_DIRECTORY, "person_" + str(category))

        if not isdir(category_directory):
            mkdir(category_directory)

        for sample_number, unprocessed_sample in enumerate(some_samples_in_a_category):
            collect()

            sample = normalize_array(compute_stft_channels(unprocessed_sample))

            #max_index = ugly_argmax(reduced_range_sample)
            #cropped_sample_frames = []
            #for reduced_range_sample_frame in np.moveaxis(reduced_range_sample, -1, 0):
            #    cropped_reduced_range_sample_frame = crop_around_row_index(reduced_range_sample_frame, max_index, int(0.5*))
            #    cropped_sample_frames.append(cropped_reduced_range_sample_frame)
            #cropped_sample = np.stack(cropped_sample_frames, axis=-1)
            # cropped_sample = np.expand_dims(cropped_sample, -1) # only need this if not using 3 channels

            if not saved_shape_once:
                Path(consts.INPUT_SHAPE_FILENAME + str(sample.shape)).touch()
                saved_shape_once = True

            np.save(join(category_directory, str(sample_number)), sample)


def crop_around_row_index(frame, index, amount):
    if index + amount > frame.shape[0]:
        return frame[-2 * amount:, :]
    elif index - amount < 0:
        return frame[0:2 * amount, :]
    else:
        return frame[index - amount:index + amount, :]


def load_raw_data(directory_path, filetype, loading_function):
    """
    Load some data and convert it to NumPy array.
    Assumes data is labeled with class at the end of filename.
    """
    print("\n\n\nLoading raw data\n\n\n")

    data_arrays = []
    classes = []
    files = listdir(directory_path)
    pattern = r'.*_.*_(\d+)\.mat'

    for filename in tqdm(files, total=len(files)):

        filepath = join(directory_path, filename)

        if filepath[-len(filetype):] == filetype:

            classes.append(int(match(pattern, filename).group(1)))
            array = loading_function(filepath)
            data_arrays.append(array)

    return data_arrays, classes


def loadmat_first_dict_radar_three_data_only(mat_filepath):
    """Specifically load only the third radar data from a .mat file"""

    return sp.io.loadmat(mat_filepath, variable_names=("hil_resha_aligned", ))["hil_resha_aligned"][:, :, 3]


def average_out_zeros(data_arrays):
    """
    If a data point is zero, replace it with 
    the average of the previous and next point.
    """

    for i, data_array in enumerate(data_arrays):

        data_array = handle_endpoints(data_array)
        neighbor_averages = (np.roll(data_array, -1, axis=1) + np.roll(data_array, 1, axis=1)) / 2
        data_arrays[i] = np.where(data_array == 0, neighbor_averages, data_array)

    return data_arrays


@jit(nopython=True)
def handle_endpoints(data_array):
    """We don't want to average endpoints if they happen to be zero"""

    if not np.all(data_array[:, 0]):
        data_array[:, 0] = np.where(data_array[:, 0] == 0, data_array[:, 1], data_array[:, 0])

    if not np.all(data_array[:, -1]):
        data_array[:, -1] = np.where(data_array[:, -1] == 0, data_array[:, -2], data_array[:, -1])

    # assert (np.all(data_array[:, 0]) and np.all(data_array[:, -1]))  # make more flexible if needed
    return data_array


def analyze_data_quality(data_arrays):
    """Compute some statistics about our data."""

    abs_mean_median_diff = np.abs(np.mean(data_arrays, axis=-1) - np.median(data_arrays, axis=-1))

    print("\n\nAvg abs(mean - median): ", np.mean(abs_mean_median_diff))
    print("\n\nMin abs(mean - median): ", np.min(abs_mean_median_diff))
    print("\n\nMax abs(mean - median): ", np.max(abs_mean_median_diff))
    print("\n\nHave NaNs?: ", np.any(np.isnan(data_arrays)))
    print("\n\nHave Infs?: ", np.any(np.isinf(data_arrays)))


def split_data_arrays_into_stacked_samples(data_arrays):
    """
    Every array in data_arrays gets split into sub arrays, where
    the last (possibly more than one) subarray might not be of correct length 
    due to variation of length in the arrays in data_arrays. 
    These subarrays are filtered out, before stacking the rest. 
    We are then left with a list of stacked subarrays, with the same length
    as data_arrays.
    """
    print("\n\n\nSplitting data arrays into stacked samples\n\n\n")

    #@jit(nopython=True)
    def split_one_array(data_array):
        return np.array_split(data_array, np.arange(consts.SAMPLE_TIMEBINS, data_array.shape[-1], consts.SAMPLE_TIMEBINS), axis=-1)

    def split_all_arrays(list_of_arrays):
        return map(split_one_array, list_of_arrays)

    #@jit(nopython=True)
    def filter_one_list_of_arrays(list_of_arrays):
        return filter(lambda x: x.shape[-1] == consts.SAMPLE_TIMEBINS, list_of_arrays)

    def filter_all_lists_of_arrays(list_of_list_of_arrays):
        return map(filter_one_list_of_arrays, list_of_list_of_arrays)

    def stack_one_list_of_arrays(list_of_arrays):
        return np.stack(list_of_arrays, axis=0)
    
    #@jit(python=True)
    def filter_out_empty(list_of_list_of_arrays):
        return filter(lambda x: x != [], list_of_list_of_arrays)

    def stack_all_lists_of_arrays(list_of_list_of_arrays):
        return map(stack_one_list_of_arrays, list_of_list_of_arrays)

    list_of_list_of_arrays = split_all_arrays(data_arrays)
    filtered_list_of_list_of_arrays = filter_out_empty(filter_all_lists_of_arrays(list_of_list_of_arrays))
    list_of_stacked_arrays = stack_all_lists_of_arrays(filtered_list_of_list_of_arrays)

    #split_array = np.stack(np.array_split(data_arrays, np.arange(Nt, data_arrays.shape[-1], Nt), axis=-1)[:-1],
    #axis=1)

    return list_of_stacked_arrays


def compute_stft_channels(unprocessed_sample):
    """Function that defines the type of STFTs we perform"""

    WINDOW_LENGTH = 172
    HOP = 5

    kwargs = {
        "return_onesided": False,
        "axis": -1,
        "scaling": "spectrum",
        "window": sp.signal.windows.hann(WINDOW_LENGTH, sym=False),
        "nperseg": WINDOW_LENGTH,
        "noverlap": WINDOW_LENGTH - HOP,
    }
    f, t, zxx = sp.signal.stft(unprocessed_sample, **kwargs)  # TODO: : set fs to 1 / 0.0082 = 122

    zxx = sp.fft.fftshift(zxx, axes=(1, ))

    ratio = float(zxx.shape[1]) / float(zxx.shape[0])
    zxx = sp.ndimage.zoom(zxx, (ratio, 1, 1), order=3, grid_mode=False)
    sample = np.stack([jit_abs(zxx), np.real(zxx), np.imag(zxx)], axis=-1)

    return sample


def ugly_argmax(reduced_range_sample):
    """
    Ugly as sin function to get the max index 
    even if numpy wants to convert 0d arrays to scalars
    """

    max_index = np.argmax(reduced_range_sample)
    max_index = max_index[0] if not np.isscalar(max_index) else max_index
    max_index = np.unravel_index(max_index, reduced_range_sample.shape)
    max_index = max_index[0] if not np.isscalar(max_index) else max_index
    return max_index


if __name__ == "__main__":
    main()
