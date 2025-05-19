import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import load_intents, prepare_data, encode_labels, build_model, save_model

def main():
    intents = load_intents('src/data/intents.json')
    patterns, labels = prepare_data(intents)

    # Convert text patterns to numeric vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns).toarray()

    # Encode labels
    y, encoder = encode_labels(labels)

    # Save vectorizer and label encoder for inference
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_model(X_train.shape[1], len(set(labels)))
    model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

    # Save model
    save_model(model, 'chatbot_model.h5')

if __name__ == "__main__":
    main()
