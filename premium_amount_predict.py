import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import nltk
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
warnings.filterwarnings('ignore')
nltk.download('stopwords')
import contractions
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from autocorrect import Speller
spell = Speller(fast=True)
import logging

logger = logging.getLogger(__name__)

tfidf_vectorizer = pickle.load(open("tfidf_premium_amount_1.pickle", "rb"))
model = pickle.load(open("svc.pkl", "rb"))



def text_clean(sentence):
    logger.info("preprocessing begins....")
    if sentence:
        try:
            sentence = sentence.lower()
            sentence = re.sub("[^a-zA-Z]", " ", str(sentence))
            sentence = sentence.split()
            sentence = [word for word in sentence if word not in stopwords]
            sentence = " ".join([ps.stem(token) for token in sentence])
            sentence = word_tokenize(sentence)
            sentence = sentence[15:200]
            return " ".join(sentence)
            return sentence
        except ValueError:
            return None
    else:
        return None



def policy_no_main(text):
    if text:
        try:
            text = text_clean(text)
            text = np.array([text])
            tfIdf = tfidf_vectorizer.transform(text)
            y_pred = model.predict(tfIdf)
            return int(y_pred[0])
        except ValueError:
            pass
    else:
        return None


if __name__ == '__main__':
    text = "  So our Saurabh is talking to you about our system was America. Yes, the company's finishing critical one form is fine, say yes, thank you for talking about the policy. Before proceeding, I would like to tell you that the call is being recorded for internal quality purposes. -Three-plus policy Double-to Six Four Three Savin Policy Come Forty Four Tha Six Hundred Thirty Nainty Ppay Do Flat D for Two Twenty and Twenty Flat JNU to Twenty is a nominee who has the latest photo of Jyoti Basic Account One once again because of your account not having spin den at the rate for the policy The premiums could not be debited, say yes, will you say yes, it is du-date, okay, is it? Was it concerned once you spoke French only? You are sitting here to do a task with yourself as to what you would do, then you mail it to one of your stores mail id for mail. Let me tell you Service at HDFC.com OK Service SERVIC Service SHDFC Dat work policy is ok ok mail and only how long will I check the balance for now? Yes, yes yes yes, millions of fans debt grace period for and Eric Zero Etty will become possible right now. If I will make my request till thirty, what will I take with you? Yes, you will get four policy benefits in the office. Your details are updated in the rest of the policy, your debit card is provided, Aadhaar card is provided.  "
    print(policy_no_main(text))