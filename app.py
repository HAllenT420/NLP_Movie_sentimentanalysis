import streamlit as st
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load saved models
with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_logreg_model = pickle.load(f)
with open('w2v_model.pkl', 'rb') as f:
    loaded_w2v_model = pickle.load(f)

def avg_word2vec(sentence, model):
    vec = []
    for word in sentence:
        if word in model.wv:
            vec.append(model.wv[word])
    if vec:
        return np.mean(np.array(vec), axis=0)
    else:
        return np.zeros(100)

def predict_sentence(sentence):
    tokenized_sentence = word_tokenize(sentence.lower())
    input_vector = avg_word2vec(tokenized_sentence, loaded_w2v_model)
    input_vector = np.array([input_vector])
    prediction = loaded_logreg_model.predict(input_vector)
    return prediction

st.title("Movie Sentiment Analysis")
st.write("Enter your review for the movie:")
input_sentence = st.text_area("")

if st.button("Submit"):
    if input_sentence:
        prediction = predict_sentence(input_sentence)
        prediction_map = {0: "Negative", 1: "Positive"}
        st.write("Prediction:", prediction_map[prediction[0]])  
    else:
        st.error("Please enter a review.")