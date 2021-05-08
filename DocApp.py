import streamlit as st
import pandas as  pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from random import randint
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from TextPreprocess import CleanText
from ReadFile import ReadPDF #, ReadTxt

# import tensorflow as tf
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def app():
	
    st.title("Document Classification App :page_with_curl:")

    # Below line will produce a dropdown to choose ML/DL model
    st.subheader('Choose Machine Learning model')
    option = st.selectbox('',
    ('Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'Random Forest'))  

    # Reading Pickle files for TFIDF and other Machine Learning models
    tfidf=joblib.load(open('TFIDF.pkl', 'rb')) 

    svm=joblib.load(open('SVM.pkl', 'rb'))
    nb= joblib.load(open('MultinomialNB.pkl', 'rb'))
    rf = joblib.load(open('RandomForest.pkl', 'rb'))
    lr = joblib.load(open('LogisticRegression.pkl', 'rb'))


    if option=='Logistic Regression':
        model= lr
    elif option=='Naive Bayes':
        model=nb
    elif option=='Support Vector Machine':
        model=lr
    elif option=='Random Forest':
        model=rf    

    # Below code will take(upload) the PDFs and Txt files as input
    st.subheader('Please select PDFs or Txt file only')
    uploaded_file = st.file_uploader("", type = ['txt', 'pdf'])
    file_text = ""
    # st.text(uploaded_file)
    if uploaded_file:
        if uploaded_file.name.endswith("pdf"):
            ReturnText_obj = ReadPDF(uploaded_file)
            file_text = ReturnText_obj.ReturnPDFText()
        elif uploaded_file.name.endswith("txt") :
            file_text = uploaded_file.read()


    # Below line will call the other code to read the PDFs using PyMuPDF/Fitz
    st.subheader("Below is text from PDF/Txt file you have uploaded, you can insert or edit it as well")   
    # Below line is to take input text from user/ or if they want to edit the extracted text from PDFs/Txt
    st.write("Your Text Here will appear below once file is uploaded. You can extend the size of box by expanding it from bottom right side!")
    userInput = st.text_area("", file_text ,height = 50,)
    
    label_dict = {0: 'Atheism',1: 'Automobile',2: 'Computer',
    3: 'Medicine',4: 'Politics',5: 'Religion',6: 'Sales',7: 'Science',8: 'Sport'}

    if st.button('Predict Class!'):
        with st.spinner("Working on your Text!"):
            CleanText_obj = CleanText(userInput)
            cleanedText = CleanText_obj.ReturnCleanText()
            transformedText=tfidf.transform([cleanedText]) 

            modelOutput=model.predict(transformedText)
            modelOutput_prob=model.predict_proba(transformedText)
            st.write("Our model is predicating this text is from-->", label_dict[modelOutput[0]])
            st.write('Below graph is showing the probability our model is giving for your text for different classes!')

        fig=go.Figure(data=[go.Bar(x = list(label_dict.values()),y=modelOutput_prob[0], marker={'color':np.arange(16)})])
        fig.update_layout(autosize=True ,plot_bgcolor='rgb(275, 275, 275)')
        fig.data[0].marker.line.width = 3
        fig.data[0].marker.line.color = "black" 
        st.plotly_chart(fig)
        st.balloons()
  

if __name__ == "__main__":
	app()
