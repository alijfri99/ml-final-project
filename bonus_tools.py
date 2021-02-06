import numpy as np
import os
import matplotlib.image as matimg
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import norm
from past.builtins import raw_input
from tensorflow.keras.models import load_model
from vae import latent2im, im2latent
from geneticalgorithm import geneticalgorithm as ga
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input


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


def genetic(encodings, category):
    cnn = DenseNet169(include_top=False, input_shape=(224, 224, 3))
    image_classifier = load_model('image_model.h5')

    def f(X):
        sum = X[0] * encodings[0]
        for i in range(1, len(X)):
            sum += X[i] * encodings[i]

        decoded_image = (decode_image(sum) * 255).astype(np.uint8)
        decoded_image = Image.fromarray(decoded_image)
        decoded_image = decoded_image.resize((224, 224))
        decoded_image = np.expand_dims(decoded_image, axis=0)
        decoded_image = preprocess_input(decoded_image)
        features = cnn.predict(decoded_image)
        answer = image_classifier.predict(features)[0][category]
        return -answer

    varbound = np.array([[0, 1]]*10)

    model = ga(function=f, dimension=10, variable_type='real', variable_boundaries=varbound)
    model.run()
    return model.output_dict['variable']


def combine_images(variables, encodings):
    cnn = DenseNet169(include_top=False, input_shape=(224, 224, 3))
    image_classifier = load_model('image_model.h5')

    sum = variables[0] * encodings[0]
    for i in range(1, len(variables)):
        sum += variables[i] * encodings[i]

    decoded_image = (decode_image(sum) * 255).astype(np.uint8)
    decoded_image = Image.fromarray(decoded_image)
    decoded_image = decoded_image.resize((224, 224))
    plt.imshow(decoded_image)
    plt.show()

    decoded_image = np.expand_dims(decoded_image, axis=0)
    decoded_image = preprocess_input(decoded_image)
    features = cnn.predict(decoded_image)
    prediction = image_classifier.predict(features)
    print("Prediction:", prediction)


def get_input():
    lines = []
    try:
        while True:
            lines.append(raw_input())
    except EOFError:
        pass
    lines = "\n".join(lines)
    return lines
