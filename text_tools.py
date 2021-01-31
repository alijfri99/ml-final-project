import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm


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
            filtered_text = [w for w in nltk.word_tokenize(text.read()) if w not in stop_words and w != '.']
            final_text = [porter.stem(word) for word in filtered_text]
            final_text = list(dict.fromkeys(final_text))
            dataset.append([final_text, class_num])

    return dataset


def split_dataset(dataset):
    dataset_x = []
    dataset_y = []
    for features, label in dataset:
        dataset_x.append(features)
        dataset_y.append(label)

    return dataset_x, dataset_y


a = load_texts("train")
b, c = split_dataset(a)
print(b)
print(len(b), len(c))
print(c)
