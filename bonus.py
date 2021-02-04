import numpy as np
from bonus_tools import *
from text_tools import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

train = load_texts("train")
train_x, train_y, train_path = split_dataset_bonus(train)
test = load_texts("test")
test_x, test_y = split_dataset(test)
train_x, test_x = extract_features(train_x, test_x)
train_x, train_y = reshape(train_x, train_y)
test_x, test_y = reshape(test_x, test_y)

model = load_model('text_model.h5')
model.evaluate(test_x, test_y)
predicted_value = np.argmax(model.predict(test_x[0].reshape(1, test_x[0].shape[0])))
print(predicted_value, get_categories()[predicted_value])
a = find_nearest_texts(test_x[0], train_x)
b = get_pictures(a, train_y, train_path)
for i in b:
    plt.figure()
    plt.imshow(i)
plt.show()
input()
a = model.predict(test_x[1].reshape(1, test_x[0].shape[0]))
print(a.argmax())
print(test[1])
print(get_categories())
input()
