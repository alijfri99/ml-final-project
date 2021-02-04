from text_tools import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt

cat = get_categories()
print(cat)
train = load_texts("train")
print(train[0])
s = "dataset/train/images/" + cat[train[0][2]] + "/" + train[0][1]
print(s)
a = plt.imread(s)
plt.imshow(a)
plt.show()
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