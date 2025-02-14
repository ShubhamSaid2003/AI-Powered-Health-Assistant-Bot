import streamlit as st
import nltk
import torch # type: ignore
from transformers import pipeline
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Load a pre-trained chatbot model for health-related queries
chatbot_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium", device=0 if torch.cuda.is_available() else -1)

# Streamlit UI
st.title("AI-Powered Health Chatbot")
st.write("Welcome! How can I assist you with your health-related concerns today?")
st.write("Chat with me about health-related topics!")

# User input
user_query = st.text_input("Enter your message:")

if user_query:
    # Tokenize input
    tokens = word_tokenize(user_query)
    st.write("Tokenized Query:", tokens)
    
    # Generate chatbot response
    response = chatbot_pipeline(user_query, max_length=100, do_sample=True)
    
    st.write("Chatbot Response:", response[0]['generated_text'])

st.write("Note: This is a basic AI chatbot and should not replace professional medical advice.")

