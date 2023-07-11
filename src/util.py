"""Stuff you just need."""
import os
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numba import jit

import consts


def create_data_loaders(data_dir,
                        batch_size,
                        validation_split=0.2,
                        seed=consts.SEED,
                        external_preprocessor_func=None):
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)
    filepaths, labels = get_filepaths_and_labels(data_dir, classes)
    train_filepaths, train_labels, val_filepaths, val_labels = shuffle_and_split_data(
        filepaths, labels, validation_split, seed)

    train_data_loader = CustomDataLoader(train_filepaths, train_labels, batch_size, num_classes,
                                         external_preprocessor_func)
    val_data_loader = CustomDataLoader(val_filepaths, val_labels, batch_size, num_classes,
                                       external_preprocessor_func)

    return train_data_loader, val_data_loader


@jit(nopython=True)
def get_filepaths_and_labels(data_dir, classes):
    filepaths = []
    labels = []

    for class_idx, class_name in enumerate(classes):

        class_path = os.path.join(data_dir, class_name)

        for filename in os.listdir(class_path):

            filepath = os.path.join(class_path, filename)
            filepaths.append(filepath)
            labels.append(class_idx)

    return filepaths, labels


def shuffle_and_split_data(filepaths, labels, validation_split, seed):
    random.seed(seed)
    combined_data = list(zip(filepaths, labels))
    random.shuffle(combined_data)

    filepaths, labels = zip(*combined_data)

    split_idx = int(len(filepaths) * (1-validation_split))

    train_filepaths = filepaths[:split_idx]
    train_labels = labels[:split_idx]
    val_filepaths = filepaths[split_idx:]
    val_labels = labels[split_idx:]

    return train_filepaths, train_labels, val_filepaths, val_labels


class CustomDataLoader(tf.keras.utils.Sequence):

    def __init__(self, filepaths, labels, batch_size, num_classes, external_preprocessor_func=None):
        self.filepaths = filepaths
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.external_preprocessor_func = external_preprocessor_func

    def __len__(self):
        return len(self.filepaths) // self.batch_size

    def __getitem__(self, idx):
        batch_filepaths = self.filepaths[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_data = []

        for filepath in batch_filepaths:

            data = np.load(filepath)  # Assuming .npy files containing 3D arrays
            batch_data.append(data)

        batch_data = np.array(batch_data)
        if self.external_preprocessor_func is not None: batch_data = self.external_preprocessor_func(batch_data)
        batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        return batch_data, batch_labels


def animate_sample(numpy_array):
    fig = plt.figure()
    im = plt.imshow(numpy_array[:, :, 0], cmap='jet')

    def update_frame(i):
        im.set_array(numpy_array[:, :, i])
        return im,

    num_frames = numpy_array.shape[2]
    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=0)
    plt.show()


@jit(nopython=True)
def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr-min_val) / (max_val-min_val)
    return normalized_arr


@jit(nopython=True)
def jit_abs(x):
    return np.abs(x)

if __name__ == "__main__":
    sample = np.load(os.path.join(consts.PROCESSED_DATA_DIRECTORY, "person_4", "1.npy"))
    print(sample[:, :, 0, 0])
    print(sample[:, :, 0, 1])
    print(sample[:, :, 0, 2])
    print(np.any(np.isinf(sample)))
    animate_sample(sample[:, :, :, 0])
    animate_sample(sample[:, :, :, 1])
    animate_sample(sample[:, :, :, 2])
