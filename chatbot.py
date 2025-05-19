from flask import Flask, request, jsonify
import os
import json
import pickle
import numpy as np
import random
from tensorflow import keras
from utils import load_intents, preprocess_input

app = Flask(__name__)

# File paths
MODEL_PATH = 'chatbot_model.h5'
VECTORIZER_PATH = 'vectorizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
INTENTS_PATH = 'src/data/intents.json'

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
model = keras.models.load_model(MODEL_PATH)

# Load vectorizer
if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer file '{VECTORIZER_PATH}' not found.")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Load label encoder
if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file '{LABEL_ENCODER_PATH}' not found.")
with open(LABEL_ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)

# Load intents
if not os.path.exists(INTENTS_PATH):
    raise FileNotFoundError(f"Intents file '{INTENTS_PATH}' not found.")
with open(INTENTS_PATH, 'r', encoding='utf-8') as file:
    intents = json.load(file)

def get_response(user_input):
    processed_input = preprocess_input(user_input, vectorizer)
    prediction = model.predict(processed_input)
    predicted_label = np.argmax(prediction)
    intent_name = encoder.inverse_transform([predicted_label])[0]

    for intent in intents['intents']:
        if intent['intent'] == intent_name:
            return random.choice(intent['responses'])

    # fallback response if intent not found
    return "Sorry, I didn't understand that."

@app.route('/')
def home():
    return "Chatbot server is running! Use the /chat endpoint with POST requests."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'response': "Invalid request. Please provide a 'message' field."}), 400
    user_input = data['message']
    response = get_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
