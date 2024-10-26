# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:13:22 2024

@author: Administrator
"""

import streamlit as st
import pickle
import re


# Define the clean_text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters, numbers, punctuation
    text = ' '.join(text.split())  # Remove extra spaces
    return text


# Load the model and vectorizer
with open('question_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit interface
st.title('Mental Health Q&A Classifier')

# Input question from user
question = st.text_input('Enter a question:')

# Option to classify either the cleaned or raw question
use_cleaned_question = st.checkbox('Use cleaned text for classification', value=True)

# Predict category
if st.button('Classify'):
    if use_cleaned_question:
        # Apply text cleaning if checkbox is selected
        processed_question = clean_text(question)
        st.write("Classifying using cleaned text:", processed_question)
    else:
        # Use the raw question if checkbox is unchecked
        processed_question = question
        st.write("Classifying using raw text:", processed_question)

    # Vectorize the input question (either cleaned or raw based on the user's choice)
    question_vector = vectorizer.transform([processed_question])

    # Make prediction
    prediction = model.predict(question_vector)

    # Display the result
    st.write(f'The question category is: {prediction[0]}')
