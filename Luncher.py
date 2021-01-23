import numpy as np
from tools import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.utils import to_categorical

train = load_images("train", 128)
train_x = []
train_y = []

for features, label, _, _ in train:
    train_x.append(features)
    train_y.append(label)

train_x = np.array(train_x).reshape(-1, 128, 128, 1)
train_y = to_categorical(train_y, 19)
train_x = train_x/255.0
print(train_x.shape)
val_x = train_x[485:]
val_y = train_y[485:]
train_x = train_x[0:485]
train_y = train_y[0:485]

print(train_x.shape, val_x.shape)

print("Loaded and normalized the training data.")
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=train_x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=train_x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(19))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=26, batch_size=64, validation_data=(val_x, val_y))
'''
'''
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
'''