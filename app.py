from flask import Flask, render_template, request, jsonify
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model and tokenizer
with open('models/text_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Flask app setup
app = Flask(__name__)

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')

    # Preprocess input
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=10, padding='pre')  # Adjust maxlen as per your model

    # Make prediction
    predicted_probs = model.predict(token_list)
    predicted_word_index = np.argmax(predicted_probs)
    predicted_word = tokenizer.index_word[predicted_word_index]

    return jsonify({'next_word': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
