from bonus_tools import *
from text_tools import *
from tensorflow.keras.models import load_model

train = load_texts("train")
train_x, train_y, train_path = split_dataset_bonus(train)
test = load_texts("test")
test_x, test_y = split_dataset(test)
train_x, test_x, vectorizer = extract_features(train_x, test_x)
train_x, train_y = reshape(train_x, train_y)
test_x, test_y = reshape(test_x, test_y)
model = load_model('text_model.h5')

input_text = get_input()
input_text = preprocess_input(input_text, vectorizer)
predicted_value = np.argmax(model.predict(input_text))
print(get_categories()[predicted_value])
nearest_texts = find_nearest_texts(input_text, train_x)
nearest_images = get_images(nearest_texts, train_y, train_path)
encodings = encode_images(nearest_images)
variables = genetic(encodings, predicted_value)
combine_images(variables, encodings)
