import os
import nltk
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer


def load_texts(data_type):
    dataset = []
    path = "dataset/" + data_type + "/sentences"
    categories = os.listdir(path)
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()

    for category in categories:
        current_path = path + "/" + category
        class_num = categories.index(category)

        for text_path in tqdm(os.listdir(current_path)):
            text = open(current_path + "/" + text_path, 'r')
            filtered_text = [porter.stem(w) for w in nltk.word_tokenize(text.read()) if w not in stop_words
                             and w != '.']
            filtered_text = list(dict.fromkeys(filtered_text))
            dataset.append([filtered_text, text_path[:len(text_path) - 4] + ".jpg", class_num])

    if data_type == "train":
        random.shuffle(dataset)
    return dataset


def split_dataset(dataset):
    dataset_x = []
    dataset_y = []
    for features, _, label in dataset:
        dataset_x.append(features)
        dataset_y.append(label)

    return dataset_x, dataset_y


def extract_features(train_x, test_x):
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)
    return train_x.toarray(), test_x.toarray()


def reshape(dataset_x, dataset_y):
    dataset_x = np.array(dataset_x)
    dataset_y = to_categorical(dataset_y, 19)
    return dataset_x, dataset_y
