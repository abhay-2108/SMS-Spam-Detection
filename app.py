from flask import Flask, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for front-end access

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('A:\\Visual Files\\Machine Learning Project\\SMS Spam Detection\\vectorizer.pkl', 'rb'))
model = pickle.load(open('A:\\Visual Files\\Machine Learning Project\\SMS Spam Detection\\model.pkl', 'rb'))

ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    if not message:
        return jsonify({'result': 'No message provided'}), 400

    # Preprocess the message
    transformed_message = transform_text(message)

    # Vectorize and predict
    vector_input = tfidf.transform([transformed_message])
    result = model.predict(vector_input)[0]

    # Return result
    if result == 1:
        return jsonify({'result': 'Spam'})
    else:
        return jsonify({'result': 'Not Spam'})

if __name__ == '__main__':
    app.run(debug=True)
