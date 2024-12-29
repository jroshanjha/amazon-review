from flask import Flask,render_template,jsonify,request, send_file
import pickle
import os
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Vectorization technique TF-IDF or CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the text data
#vectorizer = TfidfVectorizer(max_features=1000)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

stop_words = set(stopwords.words('english'))

# Application configuration:- 
app = Flask(__name__)

# Load pre-trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template("index.html")

# Preprocess the text data
def preprocess(text):
    if not isinstance(text, str): 
        return ''
    # Tokenization
    tokens = nltk.word_tokenize(text)
    #tokens = tokens.str.replace('[^\w\s]','')
    # Remove stop words
    # stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict_review():
    data = request.json
    review = data.get('review', '')

    if not review:
        return jsonify({"error": "No review provided."}), 400
    
    #review = preprocess(review)
    # Transform the input using the vectorizer
    review_vectorized = vectorizer.transform([review])

    # Predict sentiment
    prediction = model.predict(review_vectorized)

    #sentiment = "Positive" if prediction[0] >= 4  else "Negative"
    sentiment = "Positive" if prediction[0] >= 4 else ("Negative" if prediction[0]<2 else "Neutral" )
    return jsonify({
        "review": review,
        "sentiment": sentiment
    })




if __name__=="__main__":
    app.run(debug=True,port=8080)