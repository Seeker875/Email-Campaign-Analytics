#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:23:32 2018

@author: Taran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#considering null as Na values
df = pd.read_excel("L2-sample_data.xlsx",sheet_name=0,na_values='null',
                   parse_dates=['lastSeen','firstSeen'])
df.info()
df.describe()
df.shape
df.columns

df.head()
df.tail(1)
#Number of brands
df.brand.nunique()
# 57 brands
# Number of obs for each brand
#top 10 brands
df.brand.value_counts().head(10)
# NeimanMarcus has highest 4469
df.brand.value_counts().tail()

#Brands with highest null values 


# target var is readRatePercent
#null in 
df.readRatePercent.value_counts(dropna=False).head()
#1843 nan
# 1500 zero


df.brand[df.readRatePercent.isnull()].value_counts(dropna=False)

#JcPenny has only 2 missing values, Nm has 27,
df.brand[df.readRatePercent == 0].value_counts(dropna=False)
#JcPenny has 142 zero read rate percent

df.hasCreative.describe()


df.dtypes
#inbox count should be numeric

#df.inboxCount = df.inboxCount.apply(pd.to_numeric)

#inbox count should not be object type, finding non int values
icList = [i for i in df.inboxCount.unique() if type(i) !=int ]

# replacing these values
for _ in icList: 
    df.loc[df.inboxCount == _ , 'inboxCount'] = np.NaN

df.inboxCount = df.inboxCount.apply(pd.to_numeric)


df = df.drop(['subject', 'imageDownloadUrl','lang'], axis=1)

df.info()

# workin on JC

Jc = df.loc[df.brand == 'JCPenney',:]

Jc.loc[df.readRatePercent.isnull(),:]


Jc.readRatePercent.plot('hist')

Jc.readRatePercent.hist()
Jc.readRatePercent.median()
Jc.readRatePercent.mean()

df.readRatePercent.hist()
#Jc average
round(Jc.readRatePercent.mean(),3)
#0.20864599453013324
# IndustryAverage
df.readRatePercent.mean()

# group by and get mean for all brands

comp = df[df['brand'].isin(['Sears','Macys','Kohls','Nordstrom','JCPenney'])]

comp.readRatePercent.mean()

#box plot

comp.boxplot(column = 'readRatePercent', by = 'brand')

#
def boxPlot(var,y='readRatePercent',data=Jc):
    return sns.boxplot(x=var,y=y,data=data)


Jc.boxplot(column = 'readRatePercent', by = 'brand')

Jc.plot(kind = 'scatter',x='readRatePercent',y='projectedVolume')

df.plot(kind = 'scatter',x='readRatePercent',y='projectedVolume')

df.spamCount.hist()
Jc.plot(kind = 'scatter',x='readRatePercent',y='spamCount')

Jc.plot(kind="line",x='firstSeen',y='readRatePercent')

#time series 
JcTs = pd.Series(Jc.set_index('firstSeen')['readRatePercent'],index=Jc.firstSeen)

JcTs = JcTs.dropna()

JcTs.plot()

#monthly Average of readRate
round(JcTs.resample('M').mean().rename(index=lambda x: x.strftime('%B,%Y')),4)*100

#round(JcTs.resample('Q').mean(),3)
def prop(data,colBy,colWith):
    # to cal prop of proportion with target var
    return data.groupby(colBy)[colWith].mean()

prop(df,'hasCreative','readRatePercent')
prop(Jc,'hasCreative','readRatePercent')


prop(Jc,'mobileReady','readRatePercent')
prop(df,'mobileReady','readRatePercent')


prop(Jc,'isCommercial','readRatePercent')
#isCommercial true for Jc 
prop(df,'isCommercial','readRatePercent')


df.isCommercial.describe()
#same for df


prop(Jc,'personalized','readRatePercent')
# prsonalized work very well for Jc
prop(comp,'personalized','readRatePercent')
prop(df,'personalized','readRatePercent')



Jc.groupby(['personalized','mobileReady'])['readRatePercent'].mean()

Jc.groupby(['hasCreative','mobileReady'])['readRatePercent'].mean()

prop(Jc,'personalized','readDeleteRatePercent')

prop(Jc,'hasCreative','readDeleteRatePercent')


prop(Jc,'personalized','deleteRatePercent')
prop(Jc,'hasCreative','deleteRatePercent')


#imputing missing values as per personalized and hasCreative

group = Jc.groupby(['personalized','hasCreative'])

def imp(data):
    return data.fillna(data.mean())

Jc.readDeleteRatePercent = group['readDeleteRatePercent'].transform(imp)

Jc.readRatePercent = group['readRatePercent'].transform(imp)
Jc.deleteRatePercent = group['deleteRatePercent'].transform(imp)
Jc.percentOfList = group['percentOfList'].transform(imp)

Jc.readDeleteRatePercent.mean()
Jc.deleteRatePercent.mean()

Jc.fromAddress.unique()

prop(Jc,'fromAddress','readRatePercent')

Jc.groupby('fromAddress')['readRatePercent'].count()

Jc[Jc.fromAddress == 'jcpenneyacct@e.jcpenney.com'].trans.head()
#'Heres how to reset your password',Your jcp.com account has been locked

Jc.plot(kind = 'scatter',x='readRatePercent',y='projectedVolume')

#corr

df.groupby('brand')['readRatePercent'].mean().nlargest(5)

#DavidJones           0.393713
#Debijenkorf          0.315533
#LeBonMarche          0.346021


def regPlot(var,by=None,order=1):
    return sns.lmplot(x=var,y='readRatePercent',data=Jc,hue=by,order=order)

regPlot('projectedVolume','personalized')

regPlot('projectedVolume')

regPlot('subLen')

Jc.corr()
#sns.stripplot(x='personalized',y='readRatePercent',data=Jc)

def boxPlot(var,y='readRatePercent',data=Jc):
    return sns.boxplot(x=var,y=y,data=data)

boxPlot('personalized')
boxPlot('hasCreative')
boxPlot('mobileReady')

sns.pairplot(Jc)

Jc.trans.head(10)


# dropping rows with Jc account address
Jc = Jc[(Jc.fromAddress != 'jcpenneyacct@e.jcpenney.com')]



Jc.hasCreative = Jc.hasCreative.astype(int)

Jc.mobileReady = Jc.mobileReady.astype(int)
Jc.personalized = Jc.personalized.astype(int)

#Preparing data for modelling
X = Jc.loc[:,['hasCreative','mobileReady','percentOfList','personalized','trans']]


y =Jc.readRatePercent


#preparing data for pipeline
from sklearn.preprocessing import FunctionTransformer

# trans -text data
getTrans = FunctionTransformer(lambda x: x['trans'], validate=False)

# Numerics
getNums = FunctionTransformer(lambda x: x[['hasCreative', 'mobileReady','percentOfList',
                                           'personalized']], validate=False)
  
    
    
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

union = FeatureUnion(
            transformer_list = [
                ('numerics', Pipeline([
                    ('selector', getNums )
                ])),
                ('text', Pipeline([
                    ('selector', getTrans),
                    ('vectorizer', TfidfVectorizer(ngram_range=(2,3)))
                ]))
             ]
        )
                
pl = Pipeline([
        ('union', union),
        ('reg',  LinearRegression())
    ])


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)



# Fit pl to the training data
pl.fit(X_train, y_train)               
                

r2 = pl.score(X_test, y_test)
print(r2)

    
X_train, X_test, y_train, y_test = train_test_split(Jc.trans,
                                                    Jc.readRatePercent, 
                                                    random_state=42)  
    
    
pl = Pipeline([('tfidf', TfidfVectorizer(
        ngram_range=(2,3)))
        ,
        ('reg',  LinearRegression())
    ])
pl.fit(X_train, y_train)               
                

r2 = pl.score(X_test, y_test)
print(r2)    


from sklearn.model_selection import cross_val_score
    
import xgboost as xgb

("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))

pl = Pipeline([('tfidf', TfidfVectorizer(
        ))
        ,
        ("xgb_model", xgb.XGBRegressor( objective="reg:linear"))
    ])

pl.fit(X_train, y_train)               
                

r2 = pl.score(X_test, y_test)
print(r2)   

cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict("records"), y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))



gbm_param_grid = {
    'xgb_model__learning_rate': np.arange(.05,0.5 , .05),
    'xgb_model__max_depth': np.arange(3,10, 1),
    'xgb_model__n_estimators': np.arange(50, 200, 50)
}

from sklearn.model_selection import RandomizedSearchCV

# Perform RandomizedSearchCV
randomized = GridSearchCV(estimator=pl,
                          param_grid=gbm_param_grid,
                          cv=2, verbose=1)



randomized.fit(X_train, y_train) 

# Compute metrics
print(randomized.best_score_)
print(randomized.best_estimator_)


from xgboost import plot_importance
x=pl.steps[1][1].feature_importances_

fig = plt.figure(figsize=(50, 18))

fig, ax = plt.subplots(figsize=(12,18))
_ = plot_importance(pl.steps[1][1],max_num_features=10, height=0.5)
plt.show()


plt.savefig('xg.png')

pd.DataFrame(pl.steps[1][1].get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)

a=pd.DataFrame(pl.steps[1][1].feature_importances_, columns=['weights'])

pl.steps[1][1].feature_names


nE = Jc.loc[Jc.emoLen== 0,:]

prop(nE,'hasCreative','readRatePercent')


prop(nE,'mobileReady','readRatePercent')
prop(df,'mobileReady','readRatePercent')


prop(Jc,'isCommercial','readRatePercent')


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(tokens)
show_wordcloud(Samsung_Reviews_positive['Reviews'])


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(JcTs,alpha=1)