#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:19:13 2018

@author: taranpreet
"""

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet




# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

TokenBasic ='\\S+(?=\\s+)'

# Fill missing values in df.Position_Extra

vec_alphanumeric = CountVectorizer(token_pattern = TOKENS_ALPHANUMERIC,stop_words='english')

# Fit to the data
vec_alphanumeric.fit(Jc.trans)


print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:70])


# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(Jc.trans,
                                                    Jc.readRatePercent, 
                                                    random_state=42)


('tfidf', TfidfVectorizer())

('vec', CountVectorizer(token_pattern = TOKENS_ALPHANUMERIC,stop_words='english',
        ngram_range=(2,2)))



from sklearn.feature_extraction.text import TfidfVectorizer

pl = Pipeline([('tfidf', TfidfVectorizer(
        ngram_range=(2,2)))
        ,
        ('reg',  LinearRegression())
    ])

# Fit to the training data
pl.fit(X_train,y_train)
r2 = pl.score(X_test, y_test)
print(r2)

y_pred = pl.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


'''
l1_space = np.linspace(0, 1, 30)
param_grid = {'reg__l1_ratio': l1_space}


from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(pl, param_grid=param_grid , cv=5)

gs.fit(X_train,y_train)


r2 = gs.score(X_test, y_test)
print(r2)
y_pred = gs.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

for param_name in gs.best_params_:
    print(param_name,": ",gs.best_params_[param_name])

'''



text1 = " ".join(Jc.trans.str.lower())

if word not in stop_words:
    
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# first tokenize the text
tokens = nltk.word_tokenize(text1)

tokens = [word for word in tokens if word not in stop_words]

unique_tokens = set(tokens)

from nltk.tokenize import regexp_tokenize

emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"


emo = set([ emo for emo in regexp_tokenize(text1, emoji) if emo != "'"])


tokenized_lines = [regexp_tokenize(s,"\w+") for s in Jc.trans]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in Jc.trans]

[len(t) for t in Jc.trans[ :2]]

Jc['subLen']= Jc.trans.apply(len)


from collections import Counter

import string

tokens=[token.strip(string.punctuation) for token in tokens]

# remove empty tokens
tokens=[token.strip() for token in tokens if token.strip()!='']

tokens=[ token for token in tokens if len(token) >2]

# Print the 10 most common tokens
print(Counter(tokens).most_common(20))



tokens = [token for token in nltk.word_tokenize(text1.lower()) if token.isalpha()]
tokens = [token for token in tokens if token not in stopwords.words('english')]


df.imageDownloadUrl.isnull().sum()

Jc['hasImage']= np.where(Jc.imageDownloadUrl.isnull(),0,1)


prop(df,'hasImage','readRatePercent')

corr = Jc.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

Jc.readRatePercent[Jc.hasImage==0].mean()

#text with average more than 0.25
textH = " ".join(Jc.trans[Jc.readRatePercent > 0.25])

#taking only alphanumeric tokens
tokens = [token for token in nltk.word_tokenize(textH.lower()) if token.isalpha()]
tokens = [token for token in tokens if token not in stopwords.words('english')]

print(Counter(tokens).most_common(20))


textL = " ".join(Jc.trans[Jc.readRatePercent < 0.17])


def tokens(text):
    # Get most common words
    tokens = [token for token in nltk.word_tokenize(text.lower()) if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

MostComm(textL,20)

print(Counter(tokensL).most_common(20))

def diff(text1,text2):
    #Get intersection of words    
    h = set(tokens(text1))
    l = set(tokens(text2))
    return [word for word in h if word not in l]

diff(textH,textL)



textTop3 = " ".join(df.trans[df['brand'].isin(['DavidJones','LeBonMarche','Debijenkorf'])])
 
def MostComm(text,n):
    # Get most common words
    tokens = [token for token in nltk.word_tokenize(text.lower()) if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    return Counter(tokens).most_common(n)

MostComm(textTop3,20)

from nltk.tokenize import regexp_tokenize

emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"

emo = set([ emo for emo in regexp_tokenize(text1, emoji) if emo != "'"])

#Get emos
def getEmo(text):
     return(set([ emo for emo in regexp_tokenize(text, emoji) if emo != "'"]))
    
len(getEmo(textTop3))

def getEmo(text):
     return(set([ emo for emo in regexp_tokenize(text, emoji) if emo != "'"]))

def emoLen(text):
    return len(set([ emo for emo in regexp_tokenize(text, emoji) if emo != "'"]))


    
Jc['emoLen']= Jc.trans.apply(emoLen)

