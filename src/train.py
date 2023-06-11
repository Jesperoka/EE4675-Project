import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow_hub as tfhub

PRE_TRAINED_MODEL = "https://tfhub.dev/google/yamnet/1"

model = hub.load(PRE_TRAINED_MODEL)


