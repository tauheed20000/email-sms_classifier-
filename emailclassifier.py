import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK resources
nltk.download('stopwords')

# Load the spam SMS dataset
sms = pd.read_csv('Spam SMS Collection', sep='\t', names=['label','message'])
sms.drop_duplicates(inplace=True)
sms.reset_index(drop=True, inplace=True)

# Preprocess the messages
corpus = []
ps = PorterStemmer()

for i in range(0, sms.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms.message[i])
    message = message.lower()
    words = message.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in words]
    message = ' '.join(words)
    corpus.append(message)

# Vectorize the messages
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

# Prepare labels
y = pd.get_dummies(sms['label'])
y = y.iloc[:, 1].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the classifier
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# Function to predict spam
def predict_spam(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)

# Streamlit UI
st.title('EMAIL/SMS Classifier')

# Text area for user input
text_input = st.text_area('Enter your text here:', '')

# Predict button
if st.button('Predict'):
    if text_input.strip() == '':
        st.error('Please enter some text.')
    else:
        prediction = predict_spam(text_input)
        result = "Spam" if prediction == 1 else "Not Spam"
        st.success(f'The given text is: {result}')

