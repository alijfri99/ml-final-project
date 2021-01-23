from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import os
import numpy as np
import random
import math


def load_images(data_type, validation_split):
    dataset = []
    path = "dataset/" + data_type + "/images"
    categories = os.listdir(path)

    for category in categories:
        current_path = path + "/" + category
        class_num = categories.index(category)
        model = DenseNet169(include_top=False, input_shape=(224, 224, 3))

        for image_path in tqdm(os.listdir(current_path)):
            img = image.load_img(current_path + "/" + image_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            features = model.predict(img_data)
            dataset.append([features, class_num])

    print("Loaded " + data_type)
    if data_type == "train":
        random.shuffle(dataset)
        # train = dataset[0:math.ceil(validation_split*len(dataset))]
        # val = dataset[math.ceil(validation_split*len(dataset)):]
        # return train, val
    # else:
    return dataset


def split_dataset(dataset):
    dataset_x = []
    dataset_y = []
    for features, label in dataset:
        dataset_x.append(features)
        dataset_y.append(label)

    return dataset_x, dataset_y


def extract_features(dataset_x):
    model = MobileNet(include_top=False)
    result = []
    for data in tqdm(dataset_x):
        data = preprocess_input(data)
        features = model.predict(data)
        result.append(features)
    return result


def reshape(dataset_x, dataset_y):
    dataset_x = np.array(dataset_x).reshape(-1, dataset_x[0].shape[0], dataset_x[0].shape[1], dataset_x[0].shape[2],
                                            dataset_x[0].shape[3])
    dataset_y = to_categorical(dataset_y)
    return dataset_x, dataset_y


def augment(train):
    pass
