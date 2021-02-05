import numpy as np
import os
from PIL import Image
import matplotlib.image as matimg
from numpy.linalg import norm
from tensorflow.keras.models import load_model
from vae import latent2im, im2latent


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
        img = Image.fromarray(matimg.imread("dataset/train/images/" + get_categories()[np.argmax(train_y[index])] +
                                            "/" + train_path[index]))
        img = np.array(img.resize([64, 64]))
        img = img.astype('float32') / 255.
        img = img.reshape((1, np.prod(img.shape[0:])))
        print(img.shape)
        result.append(img)
    return result


def encode_images(images):
    encoder = load_model('encoder.h5')
    result = []
    for img in images:
        encoded = im2latent(encoder, img)
        result.append(encoded)
    return result


def decode_image(image):
    decoder = load_model('decoder.h5')
    decoded = latent2im(decoder, image)
    return decoded
