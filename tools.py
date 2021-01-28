from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def load_images(data_type):
    dataset = []
    path = "dataset/" + data_type + "/images"
    categories = os.listdir(path)

    for category in categories:
        current_path = path + "/" + category
        class_num = categories.index(category)
        model = DenseNet169(include_top=False, input_shape=(224, 224, 3))

        for image_path in tqdm(os.listdir(current_path)):
            img = image.load_img(current_path + "/" + image_path, target_size=(224, 224))
            if data_type == "train":
                for i in augment(img):
                    img_data = image.img_to_array(i)
                    img_data = np.expand_dims(img_data, axis=0)
                    img_data = preprocess_input(img_data)
                    features = model.predict(img_data)
                    dataset.append([features, class_num])
            else:
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                features = model.predict(img_data)
                dataset.append([features, class_num])

    print("Loaded " + data_type)
    if data_type == "train":
        random.shuffle(dataset)
    return dataset


def split_dataset(dataset):
    dataset_x = []
    dataset_y = []
    for features, label in dataset:
        dataset_x.append(features)
        dataset_y.append(label)

    return dataset_x, dataset_y


def reshape(dataset_x, dataset_y, filename):
    dataset_x = np.array(dataset_x).reshape(-1, dataset_x[0].shape[0], dataset_x[0].shape[1], dataset_x[0].shape[2],
                                            dataset_x[0].shape[3])
    dataset_y = to_categorical(dataset_y, 19)
    np.save(open(filename + ".npy", 'wb'), dataset_x)
    np.save(open(filename + "_labels.npy", 'wb'), dataset_y)


def augment(inp_image):
    result = list()
    result.append(inp_image)
    result += flip(inp_image)
    result += rotate(inp_image, 45)
    return result


def flip(inp_image):
    result = list()
    result.append(ImageOps.flip(inp_image))
    result.append(ImageOps.mirror(inp_image))
    return result


def rotate(inp_image, angle):
    result = list()
    result.append(inp_image.rotate(angle - 20))
    result.append(inp_image.rotate(angle - 10))
    result.append(inp_image.rotate(angle))
    result.append(inp_image.rotate(-angle))
    result.append(inp_image.rotate(-angle + 10))
    result.append(inp_image.rotate(-angle + 20))
    return result


def translate(inp_image, amount):
    result = list()
    result.append(inp_image.transform(inp_image.size, Image.AFFINE, (1, 0, amount, 0, 0, 0)))
    return result


def noise(inp_image):
    result = list()
    result.append(inp_image.filter(ImageFilter.GaussianBlur(radius=1)))
    result.append(inp_image.filter(ImageFilter.GaussianBlur(radius=2)))
    return result
