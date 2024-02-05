import streamlit as st 
import pickle 
import nltk
import string

# for Removing Stopwords and Punctuation
from nltk.corpus import stopwords
import string 

# for Stemming
from nltk.stem.porter import PorterStemmer
PS = PorterStemmer()

# Importing the model and tdidf

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Pre-processing the Email that has been provided

def transform_text(text):
    text = text.lower()   # Lowercasing
    text = nltk.word_tokenize(text) #Tokenizing(Breaking a sentence into words)
    
    y = []
    for i in text:
        if i.isalnum(): # Removing Special Characters 
            y.append(i)
    text = y.copy()
    #print(text)
    y.clear()
    
    for i in text:     # Removing Stopwords and punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y.copy()
    y.clear()
        
    for i in text:     # Stemming the word to root form
        y.append(PS.stem(i))
        
    
    return " ".join(y)

st.title('Email Spam Classifier')
input_email = st.text_area('Enter the Email')

if st.button('Predict'):

	# STEP 1: Transform the data(Lowrcasing, removing special characters, removing stopwords, punctuation, stemming)

	transformed_email = transform_text(input_email)

	# STEP 2: Vectorizating the Transfromed_email

	vector_input = tfidf.transform([transformed_email])

	# STEP 3: Predict using the model.

	result = model.predict(vector_input)

	if result == 1:
    		st.header('Email is SPAM')
	else:
    		st.header('Email is not a SPAM')