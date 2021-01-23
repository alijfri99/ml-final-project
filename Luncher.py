import numpy as np
from tools import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

train = load_image_features("train")
train_x = []
train_y = []

for features, label in train:
    train_x.append(features)
    train_y.append(label)

train_x = np.array(train_x).reshape(-1, train_x[0].shape[0], train_x[0].shape[1], train_x[0].shape[2],
                                    train_x[0].shape[3])
train_y = to_categorical(train_y)

test = load_image_features("test")
test_x = []
test_y = []

for features, label in test:
    test_x.append(features)
    test_y.append(label)

print(test_x[0].shape)
test_x = np.array(test_x).reshape(-1, test_x[0].shape[0], test_x[0].shape[1], test_x[0].shape[2], test_x[0].shape[3])
test_y = to_categorical(test_y)

model = Sequential()
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(19))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=4, batch_size=64, validation_split=0.1)

score = model.evaluate(test_x, test_y)
print(score)

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