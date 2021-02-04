from text_tools import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

train = load_texts("train")
train_x, train_y = split_dataset(train)
test = load_texts("test")
test_x, test_y = split_dataset(test)
train_x, test_x = extract_features(train_x, test_x)
train_x, train_y = reshape(train_x, train_y)
test_x, test_y = reshape(test_x, test_y)

'''
model = Sequential()
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(19))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1000, batch_size=32, validation_data=(test_x, test_y))
'''