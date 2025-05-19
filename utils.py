import json
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

def load_intents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    return intents

def prepare_data(intents):
    patterns = []
    labels = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            labels.append(intent['intent'])  # Use 'intent' consistently
    return patterns, labels

def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def save_model(model, model_path):
    model.save(model_path)

def preprocess_input(user_input, vectorizer):
    return vectorizer.transform([user_input]).toarray()
