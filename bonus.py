from text_tools import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

train = load_texts("train")
train_x, train_y = split_dataset(train)
test = load_texts("test")
test_x, test_y = split_dataset(test)
train_x, test_x = extract_features(train_x, test_x)
train_x, train_y = reshape(train_x, train_y)
test_x, test_y = reshape(test_x, test_y)

model = load_model('text_model.h5')
model.evaluate(test_x, test_y)
a = model.predict(test_x[1].reshape(1, test_x[0].shape[0]))
print(a.argmax())
print(test[1])
print(get_categories())
input()
