from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import os
import numpy as np
import random
import math


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
    dataset_y = to_categorical(dataset_y, 19)
    return dataset_x, dataset_y


def augment(train):
    pass


img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'dataset/train/images'
validation_data_dir = 'dataset/test/images'
nb_train_samples = 532
nb_validation_samples = 380
epochs = 50
batch_size = 16


def create_labels(nb_categories, nb_examples):
    train_labels = []
    for i in range(nb_categories):
        for _ in range(nb_examples):
            train_labels.append(i)

    return to_categorical(np.array(train_labels), nb_categories)


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rotation_range=20,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 brightness_range=[0.8, 1.2],
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 preprocessing_function=preprocess_input)
    model = ResNet152(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    print('Predicted!!!!!')


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = create_labels(19, 28)
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = create_labels(19, 20)

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(19, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size, validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
