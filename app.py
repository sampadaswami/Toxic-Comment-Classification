import streamlit as st
import pickle
import re

# Step 1: Load the trained model
with open('toxic_comment_model_xgb.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Step 2: Text Preprocessing - Clean the comments (same as in training)
def clean_text(text):
    # Remove non-alphabetic characters and lower the text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

# Step 3: Function to predict toxicity for user input
def predict_toxicity(user_input):
    # Preprocess the user input
    user_input_cleaned = clean_text(user_input)
    
    # Predict toxicity
    result = pipeline.predict([user_input_cleaned])[0]
    
    # Output the predictions
    prediction_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result_dict = dict(zip(prediction_labels, result))
    
    return result_dict

# Streamlit Interface
st.title("Toxic Comment Classification")
st.write("Enter a comment to analyze:")

# User input
user_input = st.text_area("Comment:", value="", placeholder="Type your comment here...")

if st.button("Analyze"):
    if user_input.strip():
        # Predict toxicity
        result = predict_toxicity(user_input)
        
        # Display results
        st.subheader("Comment Analysis Results:")
        for label, score in result.items():
            st.write(f"**{label.capitalize()}**: {'Yes' if score == 1 else 'No'}")
    else:
        st.error("Please enter a valid comment!")
