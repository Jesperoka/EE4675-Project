"""Preprocesing of data"""
from os import getcwd, scandir
from os.path import join
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage

SAMPLE_TIME = 0.0082  # sample time in seconds
MIN_RANGE = 1  # minimum range in meters
MAX_RANGE = 4.8  # maximum range in meters


def main():
    RAW_DATA_DIRECTORY = join(getcwd(), "..", "raw_data")
    PROCESSED_DATA_DIRECTORY = join(getcwd(), "..", "processed_data")

    print("\n\n\nProcessing the data in", RAW_DATA_DIRECTORY, ". . .\n\n\n")

    preprocess(RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY)

    print("\n\n\nDone! Processed data is in", PROCESSED_DATA_DIRECTORY,
          "\n\n\n")


def preprocess(input_directory_path, output_directory_path):
    """
    Load, analyze and preprocess all data in the given input directory,
    and output the processed data to the given output directory.
    """

    data_arrays, classes = load_raw_data(
        input_directory_path, ".mat", loadmat_first_dict_radar_three_data_only)

    data_arrays = average_out_zeros(data_arrays)

    #data_arrays = format_radar_data(data_arrays) # use real-valued decibel signal or raw complex signal?
    data_arrays = np.sum(data_arrays, axis=-2)

    analyze_data_quality(data_arrays)

    split_data_arrays = split_into_two_second_intervals(data_arrays)

    zxx0, zxx1, zxx2 = compute_STFTs(split_data_arrays)

    zxx1_real, zxx1_imag = resize_and_preserve_complex(zxx1, zxx0.shape)
    zxx2_real, zxx2_imag = resize_and_preserve_complex(zxx2, zxx0.shape)

    plot_random_sample(zxx0, zxx1_real, zxx1_imag, zxx2_real, zxx2_imag)

    stacked_spectrums = stack_spectrums(np.real(zxx0), np.imag(zxx0),
                                        zxx1_real, zxx1_imag, zxx2_real,
                                        zxx2_imag)

    classes = np.tile(np.array(classes), stacked_spectrums.shape[1])

    np.savez(join(output_directory_path, "stacked_spectrums"),
             stacked_spectrums=np.reshape(stacked_spectrums,
                                          (-1, *stacked_spectrums.shape[2:]),
                                          order="F"),
             classes=classes)


def load_raw_data(directory_path, filetype, loading_function):
    """
    Load some data and convert it to NumPy array.
    Assumes data is labeled with class at the end of filename.
    """

    data_arrays = []
    classes = []
    min_length = float("Inf")

    for filename in scandir(directory_path):

        if filename.path[-len(filetype):] == filetype:

            classes.append(filename.path[-(len(filetype) + 1)])
            array = loading_function(filename.path)
            data_arrays.append(array)

            if max(array.shape) < min_length: min_length = max(array.shape)

    data_arrays = [array[:, 0:min_length] for array in data_arrays]

    return np.moveaxis(np.dstack(data_arrays), -1, 0), classes


def loadmat_first_dict_radar_three_data_only(mat_filepath):
    """Specifically load only the third radar data from a .mat file"""

    return sp.io.loadmat(mat_filepath)["hil_resha_aligned"][:, :, 3]


def average_out_zeros(data_arrays):
    """
    If a data point is zero, replace it with 
    the average of the previous and next point.
    """

    for i, data_array in enumerate(data_arrays):

        data_array = handle_endpoints(data_array)
        neighbor_averages = (np.roll(data_array, -1, axis=1) +
                             np.roll(data_array, 1, axis=1)) / 2
        data_arrays[i] = np.where(data_array == 0, neighbor_averages,
                                  data_array)

    return data_arrays


def handle_endpoints(data_array):
    """We don't want to average endpoints if they happen to be zero"""

    if not np.all(data_array[:, 0]):
        data_array[:, 0] = np.where(data_array[:, 0] == 0, data_array[:, 1],
                                    data_array[:, 0])

    if not np.all(data_array[:, -1]):
        data_array[:, -1] = np.where(data_array[:, -1] == 0, data_array[:, -2],
                                     data_array[:, -1])

    assert (np.all(data_array[:, 0])
            and np.all(data_array[:, -1]))  # make more flexible if needed
    return data_array


def format_radar_data(data_arrays):  # TODO: rename
    """Same formatting as done in dataread.mlx"""

    return 20 * np.log10(np.abs(data_arrays))


def resize_and_preserve_complex(complex_array, shape):
    """
    Convenience helper to avoid casting away
    complex part when using skimage.transform.resize
    """

    real = np.real(complex_array)
    imag = np.imag(complex_array)

    common_kwargs = {
        "order": 1,
        "preserve_range": True,
        "anti_aliasing": False
    }

    real = skimage.transform.resize(real, shape, **common_kwargs)
    imag = skimage.transform.resize(imag, shape, **common_kwargs)

    return real, imag


def analyze_data_quality(data_arrays):
    """Compute some statistics about our data."""

    abs_mean_median_diff = np.abs(
        np.mean(data_arrays, axis=-1) - np.median(data_arrays, axis=-1))

    print("\n\nAvg abs(mean - median): ", np.mean(abs_mean_median_diff))
    print("\n\nMin abs(mean - median): ", np.min(abs_mean_median_diff))
    print("\n\nMax abs(mean - median): ", np.max(abs_mean_median_diff))
    print("\n\nHave NaNs?: ", np.any(np.isnan(data_arrays)))
    print("\n\nHave Infs?: ", np.any(np.isinf(data_arrays)))


def split_into_two_second_intervals(data_arrays):
    return np.moveaxis(
        np.dstack(
            np.array_split(data_arrays,
                           np.arange(243, data_arrays.shape[-1], 243),
                           axis=-1)[:-1]), -1, -2)


def compute_STFTs(split_data_arrays):
    """Wrapper that defines the type of STFTs we perform"""

    common_kwargs = {
        "window": "hann",
        "return_onesided": False,
        "axis": -1,
        "scaling": "spectrum"
    }

    _, _, zxx0 = sp.signal.stft(split_data_arrays,
                                nperseg=128,
                                noverlap=127,
                                **common_kwargs)

    _, _, zxx1 = sp.signal.stft(split_data_arrays,
                                nperseg=64,
                                noverlap=63,
                                **common_kwargs)

    _, _, zxx2 = sp.signal.stft(split_data_arrays,
                                nperseg=32,
                                noverlap=31,
                                **common_kwargs)

    return zxx0, zxx1, zxx2


def plot_random_sample(zxx0, zxx1_real, zxx1_imag, zxx2_real, zxx2_imag):
    """Sanity check"""

    random_sample = (randint(0,
                             zxx0.shape[0] - 1), randint(0, zxx0.shape[1] - 1))

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].pcolormesh(np.real(zxx0[random_sample]))
    axs[1, 0].pcolormesh(zxx1_real[random_sample])
    axs[2, 0].pcolormesh(zxx2_real[random_sample])
    axs[0, 1].pcolormesh(np.imag(zxx0[random_sample]))
    axs[1, 1].pcolormesh(zxx1_imag[random_sample])
    axs[2, 1].pcolormesh(zxx2_imag[random_sample])
    plt.show()


def stack_spectrums(zxx0_real, zxx0_imag, zxx1_real, zxx1_imag, zxx2_real,
                    zxx2_imag):
    """Create 6 channel image of different spectrums"""

    return np.stack([
        zxx0_real,
        zxx0_imag,
        zxx1_real,
        zxx1_imag,
        zxx2_real,
        zxx2_imag,
    ],
                    axis=-1)


if __name__ == "__main__":
    main()
