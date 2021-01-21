import tensorflow as tf

from Controllers import ImagesController
from Controllers import CaptionsController

train_dir = "dataset/train/images"
test_dir = "dataset/test/images"


def read_images(mode):   # mode : test , train
    if mode == 'train':
        return tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=(ImagesController.img_height, ImagesController.img_width),
            batch_size=ImagesController.batch_size)
    elif mode == 'test':
        return tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            image_size=(ImagesController.img_height, ImagesController.img_width),
            batch_size=ImagesController.batch_size)


def read_captions():
    pass
