import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load SVM model
svm_model = joblib.load('svm_model_tfidf.joblib')

# Load TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Mapping between encoded labels and actual tag names
tag_names = {0: 'android', 1: 'c#', 2: 'c++', 3: 'html', 4: 'ios', 5: 'java', 6: 'javascript', 7: 'jquery', 8: 'php', 9: 'python'}

# Streamlit app
st.title('Coding Question Tag Classifier')
st.write('Enter your coding-related question below:')

# Input text area for user's question
user_question = st.text_area('Input your question here:', '')

# Preprocess user input
preprocessed_question = preprocess_text(user_question)

if st.button('Classify'):
    if user_question.strip() == '':
        st.error('Please enter a question.')
    else:
        # Vectorize the preprocessed question
        question_vectorized = tfidf_vectorizer.transform([preprocessed_question])
        # Predict the tag
        predicted_tag_encoded = svm_model.predict(question_vectorized)[0]
        predicted_tag = tag_names.get(predicted_tag_encoded, 'Unknown')
        st.success(f'Predicted Tag: {predicted_tag}')