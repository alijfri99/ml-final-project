from tools import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

'''
train = load_images("train")
test = load_images("test")
train_x, train_y = split_dataset(train)
test_x, test_y = split_dataset(test)
reshape(train_x, train_y, "train")
reshape(test_x, test_y, "test")
'''


train_x = np.load(open("train.npy", 'rb'))
train_y = np.load(open("train_labels.npy", 'rb'))
test_x = np.load(open("test.npy", 'rb'))
test_y = np.load(open("test_labels.npy", 'rb'))

model = Sequential()
model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(19))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1000, batch_size=64, validation_data=(test_x, test_y))
