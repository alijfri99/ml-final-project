import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers


from Utilities import FilesHandling

train_ds = FilesHandling.read_images("train")
test_ds = FilesHandling.read_images("test")

class_names = train_ds.class_names
print(class_names)
print("trains : " + str(tf.data.experimental.cardinality(train_ds).numpy()))
print("tests : " + str(tf.data.experimental.cardinality(test_ds).numpy()))

num_classes = 19

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(19))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=10
)
