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



data1 = pd.read_csv(r'C:\Users\thonnatig.HDFCSLDM.005\Downloads\true_label_run_600.csv')
data2 = pd.read_csv(r'C:\Users\thonnatig.HDFCSLDM.005\Downloads\True_Label_Data.csv')
data2 = data2[['Audio','person','premium_amount']]
data = data2.merge(data1[['application_no','translated_agent_text','transcribe_agent_text']],left_on = 'Audio',right_on = 'application_no')

data_ma = data[data['premium_amount'] == 'y']
data_mi = data[data['premium_amount'] == 'n']

df_minority_upsampled = resample(data_mi,
                             replace=True,
                             n_samples=253,
                             random_state=123)
df = pd.concat([data_ma, df_minority_upsampled])
print(df)


def text_clean(sentence):
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



df['text'] = df['translated_agent_text'].apply(text_clean)

df['premium_amount'] = df['premium_amount'].replace('y',1).replace('n',0)


tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', max_features=1600)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['premium_amount'], test_size=0.30,
                                                        stratify=df['premium_amount'], random_state=883)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


lr=LogisticRegression(solver='liblinear',penalty='l1')
svc=SVC(kernel='sigmoid',gamma=0.9)



def model(clf):
    clf.fit(X_train, y_train)
    print('Training\nReport')
    y_pred = clf.predict(X_train)
    print(classification_report(y_train, y_pred))

    print('Testing\nReport')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

model(svc)

pickle.dump(tfidf, open("tfidf_premium_amount_1.pickle","wb"))
pickle.dump(svc, open('svc.pkl','wb'))

