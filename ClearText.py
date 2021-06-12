import re
#nltk.download('stopwords')
from nltk.corpus import stopwords
#stopwords.words('english')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer
from math import log
import datetime as dt

#print (stopwords.words() [620:680])
lemmatizer = WordNetLemmatizer()
snowball = SnowballStemmer('english')

#попередня обробка тексту
def ClearText(text):
    #переведення до нижнього регістру всіх слів
    cleartext = text.lower()
    #прибирання пустих рядків та розрив рядків
    cleartext = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', cleartext) 
    #залишаємо лише слова, прибираємо пунктуацію та числа
    cleartext = re.sub("[^a-z-']", " ", cleartext)
    cleartext = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) 
    cleartext = cleartext.replace("\\", "")
    #прибираємо зайві пробіи
    cleartext = re.sub(" +", " ", cleartext)
    #ділимо речення на список слів, розбиваємо по пробілам
    cleartext = re.split(" ", cleartext)
    #прибираємо стопслова
    cleartext = [word for word in cleartext if word not in stopwords.words('english')]
    #стемінг слів
    cleartext = [snowball.stem(word) for word in cleartext]
    #лематизація слів
    cleartext = [lemmatizer.lemmatize(word) for word in cleartext]
    #прибираємо слова, довжина який менше 3 букв
    cleartext = [word for word in cleartext if len(word) > 3]
    return cleartext

import pandas as pd
df_train_2 = pd.read_csv('train_news.csv')
df_train_2.columns = ['Category', 'Title', 'Text']
df_test_2 = pd.read_csv('test_news.csv')
df_test_2.columns = ['Category', 'Title', 'Text']

df_train_2['ClearText'] = df_train_2.apply(lambda x: ClearText(x['Text']), axis=1)
df_test_2['ClearText'] = df_test_2.apply(lambda x: ClearText(x['Text']), axis=1)
