#from inspect import getfullargspec
import os
from pprint import pprint

#import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
#import tensorflow_hub as tfhub
#from classification_models_3D.tfkeras import Classifiers
from official.projects.movinet.modeling import movinet, movinet_model

import consts
import util

os.environ['CUDA_VISIBLE_DEVICES'] = ""

#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
#ks.mixed_precision.set_global_policy("mixed_float16")
#ks.backend.set_floatx("float16")
BATCH_SIZE = 5 
NUM_CLASSES = 4
VALIDATION_SPLIT = 0.2
INPUT_SHAPE = consts.INPUT_SHAPE
assert (INPUT_SHAPE is not None)

training_data_loader, validation_data_loader = util.create_data_loaders(consts.PROCESSED_DATA_DIRECTORY,
                                                                        BATCH_SIZE, VALIDATION_SPLIT, consts.SEED,
                                                                        None)

# TODO: test this again
#PretrainedModel, preprocess_input = Classifiers.get("convnext_tiny")
#pretrained_model = PretrainedModel(include_top=False, input_shape=(38, 128, 128, 3), weights="imagenet")
#for layer in pretrained_model.layers[:len(pretrained_model.layers)-0]:
#    layer.trainable = False

#¤hub_url = "https://tfhub.dev/shoaib6174/swin_base_patch244_window877_kinetics600_22k/1"
#hub_url = "https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/3"
#encoder = tfhub.KerasLayer(hub_url, trainable=True, input_shape=(3, 38, 128, 128))
#inputs = tf.keras.layers.Input(shape=[None, None, None, 3], dtype=tf.float32, name='image')
#outputs = encoder(dict(image=inputs))
#pretrained_model = ks.Model(inputs, outputs, name="movinet")

#model_path = "C:\\Users\\jespe\\Documents\\TUDelft\\EE4675_OCR\\swin_tiny_patch244_window877_kinetics400_1k_1"
#pretrained_model = ks.models.load_model(model_path, compile=True)

#for layer in pretrained_model.layers:
#    layer.trainable = False

# Model Architechture
backbone = movinet.Movinet(model_id="a3",
                           causal=True,
                           conv_type="2plus1d",
                           se_type="2plus3d",
                           activation="hard_swish",
                           gating_activation="hard_sigmoid",
                           use_positional_encoding=True,
                           use_external_states=True)

# Temporary model to restore checkpoint weights
temp_model = movinet_model.MovinetClassifier(backbone, num_classes=600, output_states=True)
movi_input = tf.ones([BATCH_SIZE, 13, 172, 172, 3])
temp_model.build(movi_input)

movi_init_states = temp_model.init_states(tf.shape(movi_input))

# Restore checkpoint weights
checkpoint_path = tf.train.latest_checkpoint("C:\\Users\\jespe\\Documents\\TUDelft\\EE4675_OCR\\EE4675-Project\\models\\movinet_a3_stream")
checkpoint = tf.train.Checkpoint(model=temp_model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

# Create custom classifier for our dataset
def build_classifier(sample_shape, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes,
      output_states=False)
  model.build(sample_shape)

  return model

stream_model = build_classifier([BATCH_SIZE, *INPUT_SHAPE], backbone, NUM_CLASSES)


#pretrained_model.trainable = False
#pprint(pretrained_model.__dict__)
#pretrained_model.layers.pop(0) # remove the head

#stream_model.summary()
#input("wait")

i = 0
for layer in stream_model.layers[:-1]:
    layer.trainable = False 
    #print(i, layer.name)
    #i += 1


#pretrained_model.layers[-1].activation = ks.activations.linear

#pretrained_model.inputs["image"] = my_input


video = ks.layers.Input(shape=consts.INPUT_SHAPE)
frames_first_video = ks.layers.Permute((3, 1, 2, 4))(video)

output = stream_model({**movi_init_states, "image": frames_first_video})

#base = pretrained_model.layers[-1].output

#X = ks.layers.Dense(1024, activation="relu", name="myDense")(base)

#X = ks.layers.GaussianNoise(stddev=0.001)(X)
#X = ks.layers.Dropout(0.3)(classifier_input)

#classifier_output = ks.layers.Dense(NUM_CLASSES, activation="softmax", name="SoftmaxOutput")(X)

model = ks.Model(inputs=video, outputs=output)

#model = ks.Sequential([
#    ks.layers.Input(shape=consts.INPUT_SHAPE, name="Input"),
#    #ks.layers.Permute((4, 1, 2, 3)),
#    ks.layers.Flatten(),
#    #ks.layers.GlobalAveragePooling3D(),
#    ks.layers.Dense(2048, activation="relu"),
#    ks.layers.Dense(NUM_CLASSES, activation='softmax', name="SoftmaxOutput")
#])

#model.layers[-3].activation = ks.activations.linear
#print(model.layers[-3].name)
#model = pretrained_model


model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

#print(pretrained_model.inputs)

#print(model.layers[-3].activation)


#input("\n\nPress any key to train\n\n")

history = model.fit(
    x=training_data_loader,
    validation_data=validation_data_loader,
    batch_size=BATCH_SIZE,
    epochs=20,
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
