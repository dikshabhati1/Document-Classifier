import nltk
import re


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 

lm=WordNetLemmatizer()

class CleanText():
    def __init__(self, userInput):
        self.text = userInput

    def ReturnCleanText(self):
        # change the text into lower case.(Note: in case of social media text, it is good to leave them as it is!)
        text = self.text.lower()
        # removing xml tags from tweets
        text =BeautifulSoup(text, 'lxml').get_text()
        # removing URLS 
        text =re.sub('https?://[A-Za-z0-9./]+','',text)
        # removing words with "@"
        text =re.sub(r'@[A-Za-z0-9]+','',text)
        # removing special characters
        text = re.sub(r"\W+|_", ' ', text)
        # tokenization of sentences
        text = word_tokenize(text)
        # lemmatize the text using WordNetn
        words = [lm.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]   
        return " ".join(words)