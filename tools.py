import cv2
import os
import random


def load_images(data_type, image_size):
    dataset = []
    path = "dataset/" + data_type + "/images"
    categories = os.listdir(path)
    for category in categories:
        current_path = path + "/" + category
        class_num = categories.index(category)

        for img in os.listdir(current_path):
            img_array = cv2.imread(current_path + "/" + img, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (image_size, image_size))
            dataset.append([img_array, class_num, category, img])

    random.shuffle(dataset)
    return dataset
