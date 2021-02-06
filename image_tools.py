from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from PIL import ImageOps, ImageChops, ImageFilter
import os
import numpy as np
import random
import copy


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
    result += noise(inp_image)
    result.append(equalize(inp_image))
    result.append(reduce(inp_image))
    result += shift(inp_image, 30)
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


def noise(inp_image):
    result = list()
    result.append(inp_image.filter(ImageFilter.GaussianBlur(radius=1)))
    result.append(inp_image.filter(ImageFilter.GaussianBlur(radius=2)))
    # result.append(salt_and_pepper(inp_image, 0.05))
    # result += channel_noise(inp_image)
    return result


def salt_and_pepper(inp_image, amount):
    result = copy.copy(inp_image)
    pixels = result.load()
    area = amount * result.size[0] * result.size[1] * 0.5
    for i in range(int(area)):
        xb = random.randint(0, result.size[0] - 1)
        yb = random.randint(0, result.size[1] - 1)
        xw = random.randint(0, result.size[0] - 1)
        yw = random.randint(0, result.size[1] - 1)
        pixels[xb, yb] = (0, 0, 0)
        pixels[xw, yw] = (255, 255, 255)
    return result


def channel_noise(inp_image):
    result = list()

    r = (1, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0)
    g = (0, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 0, 0)
    b = (0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 1, 0)
    rg = (1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 0, 0)
    rb = (1, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 1, 0)
    gb = (0, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0)

    result.append(inp_image.convert("RGB", r))
    result.append(inp_image.convert("RGB", g))
    result.append(inp_image.convert("RGB", b))
    result.append(inp_image.convert("RGB", rg))
    result.append(inp_image.convert("RGB", rb))
    result.append(inp_image.convert("RGB", gb))
    return result


def equalize(inp_image):
    result = ImageOps.equalize(inp_image)
    return result


def reduce(inp_image):
    result = ImageOps.posterize(inp_image, 2)
    return result


def shift(inp_image, amount):
    result = list()
    result.append(ImageChops.offset(inp_image, -amount, 0))
    result.append(ImageChops.offset(inp_image, amount, 0))
    result.append(ImageChops.offset(inp_image, 0, -amount))
    result.append(ImageChops.offset(inp_image, 0, amount))
    result.append(ImageChops.offset(inp_image, -amount, -amount))
    result.append(ImageChops.offset(inp_image, -amount, amount))
    result.append(ImageChops.offset(inp_image, amount, -amount))
    result.append(ImageChops.offset(inp_image, amount, amount))
    return result
