from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import numpy as np
import random
from tqdm import tqdm


def load_image_features(data_type):
    dataset = []
    path = "dataset/" + data_type + "/images"
    categories = os.listdir(path)
    model = VGG16(include_top=False)

    for category in categories:
        current_path = path + "/" + category
        class_num = categories.index(category)

        for image_path in tqdm(os.listdir(current_path)):
            img = image.load_img(current_path + "/" + image_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            features = model.predict(img_data)
            dataset.append([features, class_num])

    random.shuffle(dataset)
    return dataset
