import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import norm


def get_categories():
    return os.listdir("dataset/train/images")


def split_dataset_bonus(dataset):
    dataset_x = []
    dataset_y = []
    dataset_path = []
    for features, path, label in dataset:
        dataset_x.append(features)
        dataset_y.append(label)
        dataset_path.append(path)

    return dataset_x, dataset_y, dataset_path


def cosine_sim(a, b):
    return np.dot(a, b)/(norm(a) * norm(b))


def find_nearest_texts(text, train_x):
    result = []
    for i in range(train_x.shape[0]):
        result.append([cosine_sim(text, train_x[i]), i])
    return sorted(result, key=lambda x: x[0], reverse=True)[0:10]


def get_images(nearest_texts, train_y, train_path):
    result = []
    for _, index in nearest_texts:
        print(get_categories()[np.argmax(train_y[index])])
        result.append(plt.imread("dataset/train/images/" + get_categories()[np.argmax(train_y[index])] +
                                 "/" + train_path[index]))
    return result

