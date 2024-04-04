import streamlit as st
import pickle
import nltk
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open('new_model.pkl', 'rb'))
vectorizer = pickle.load(open('vector.pkl', 'rb'))

# Function to clean resume text
def clean_resume(text):
    # Remove URLs, hashtags, mentions, special characters, and numbers
    text = re.sub(r'http\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    # Convert to lowercase and remove extra whitespaces
    text = text.lower().strip()
    return text

def main():
    st.title("Resume Classification App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'docx', 'doc', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeError:
            resume_text = resume_bytes.decode('latin-1')

        # Clean the resume text
        cleaned_resume = clean_resume(resume_text)
        # Vectorize the cleaned resume text
        input_feature = vectorizer.transform([cleaned_resume])

        try:
            # Make predictions using the trained model
            prediction_id = model.predict(input_feature)[0]  # <-- Ensure input_feature is an array
            category_mapping = {
                0: 'Peoplesoft Developer',
                1: 'ReactJs Developer',
                2: 'SQL Developer',
                3: 'Workday'
            }
            # Output the predicted category
            if prediction_id in category_mapping:
                category_name = category_mapping[prediction_id]
                st.write("Predicted Category : ", category_name)
            else:
                st.write('Unknown Category')

            # Debugging: Output the prediction ID and cleaned resume text
            st.write("Prediction ID: ", prediction_id)
            st.write("Cleaned Resume Text: ", cleaned_resume)
        except Exception as e:
            st.error(f"Error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
