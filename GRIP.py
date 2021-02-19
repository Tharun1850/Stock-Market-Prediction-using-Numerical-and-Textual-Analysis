#!/usr/bin/env python
# coding: utf-8

# # TASK 7: Stock Market Prediction using Numerical and Textual Analysis

# Tharun Kumar Bandaru
# 
# 

# bandarutharun185@gmail.com

# Datasets used
# 

# Historical stock prices:
# :https://finance.yahoo.com/quote/IBM/history?p=IBM

# Textual News Headlines: https://bit.ly/36fFPI6
# 

# In[ ]:





# In[ ]:





# # Importing Required Datasets 

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Dense, Activation

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, metrics


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# Reading Data

# In[45]:


#reading the datasets into pandas
stock_price = pd.read_csv('/Users/tharunbandaru/Desktop/IBM.csv')
stock_headlines = pd.read_csv('/Users/tharunbandaru/Desktop/india-news-headlines.csv')


# In[5]:


stock_price.head()


# In[6]:


stock_price.head()


# In[7]:


stock_price.isna().any() 


# In[8]:


stock_headlines.isna().any()


# In[9]:


# dropping duplicates
stock_price = stock_price.drop_duplicates()

# coverting the datatype of column 'Date' from type object to type 'datetime'
stock_price['Date'] = pd.to_datetime(stock_price['Date']).dt.normalize()

# filtering the important columns required
stock_price = stock_price.filter(['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])

# setting column 'Date' as the index column
stock_price.set_index('Date', inplace= True)

# sorting the data according to the index i.e 'Date'
stock_price = stock_price.sort_index(ascending=True, axis=0)
stock_price


# In[10]:


# dropping duplicates
stock_headlines = stock_headlines.drop_duplicates()

# coverting the datatype of column 'Date' from type string to type 'datetime'
stock_headlines['publish_date'] = stock_headlines['publish_date'].astype(str)
stock_headlines['publish_date'] = stock_headlines['publish_date'].apply(lambda x: x[0:4]+'-'+x[4:6]+'-'+x[6:8])
stock_headlines['publish_date'] = pd.to_datetime(stock_headlines['publish_date']).dt.normalize()

# filtering the important columns required
stock_headlines = stock_headlines.filter(['publish_date', 'headline_text'])

# grouping the news headlines according to 'Date'
stock_headlines = stock_headlines.groupby(['publish_date'])['headline_text'].apply(lambda x: ','.join(x)).reset_index()

# setting column 'Date' as the index column
stock_headlines.set_index('publish_date', inplace= True)

# sorting the data according to the index i.e 'Date'
stock_headlines = stock_headlines.sort_index(ascending=True, axis=0)
stock_headlines


# concatenation of 2 datsets

# In[11]:


stock_data = pd.concat([stock_price, stock_headlines], axis=1)

# dropping the null values if any
stock_data.dropna(axis=0, inplace=True)

# displaying the combined stock_data
stock_data


# In[12]:


# adding empty sentiment columns 
stock_data['compound'] = ''
stock_data['negative'] = ''
stock_data['neutral'] = ''
stock_data['positive'] = ''
stock_data.head()


# In[14]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata

# Sentiment Analyzer as sid
sid = SentimentIntensityAnalyzer()

# calculating sentiment scores
stock_data['compound'] = stock_data['headline_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
stock_data['negative'] = stock_data['headline_text'].apply(lambda x: sid.polarity_scores(x)['neg'])
stock_data['neutral'] = stock_data['headline_text'].apply(lambda x: sid.polarity_scores(x)['neu'])
stock_data['positive'] = stock_data['headline_text'].apply(lambda x: sid.polarity_scores(x)['pos'])


# In[15]:


stock_data


# In[16]:


# dropping the 'headline_text' 
stock_data.drop(['headline_text'], inplace=True, axis=1)
stock_data = stock_data[['Close', 'compound', 'negative', 'neutral', 'positive', 'Open', 'High', 'Low', 'Volume']]
stock_data.head()


# In[17]:


stock_data.to_csv('stock_data.csv')


# In[18]:


stock_data = pd.read_csv('stock_data.csv', index_col = False)

# renaming the column
stock_data.rename(columns={'Unnamed: 0':'Date'}, inplace = True)

# setting the column 'Date' as the index column
stock_data.set_index('Date', inplace=True)

# displaying the stock_data
stock_data.head()


# In[19]:


stock_data.isna().any()


# In[20]:


stock_data.info()


# In[21]:


# setting figure size
plt.figure(figsize=(12,12))

# plotting close price
stock_data['Close'].plot()
plt.xlabel('Date')
plt.ylabel('Close Price ($)')


# In[24]:


stock_data.rolling(7).mean()


# In[25]:


plt.figure(figsize=(12,12))
stock_data['Close'].plot()
stock_data.rolling(window=30).mean()['Close'].plot()


# In[26]:


stock_data


# In[ ]:


DATA PREPERATION


# In[28]:


percentage_of_data = 1.0
data_to_use = int(percentage_of_data*(len(stock_data)-1))

# using 80% of data for training
train_end = int(data_to_use*0.8)
total_data = len(stock_data)
start = total_data - data_to_use

#train and test datasets
print( train_end)
print( total_data - train_end)


# In[29]:



# predicting one step ahead
steps_to_predict = 1

# capturing data to be used for each column
close_price = stock_data.iloc[start:total_data,0] #close
compound = stock_data.iloc[start:total_data,1] #compound
negative = stock_data.iloc[start:total_data,2] #neg
neutral = stock_data.iloc[start:total_data,3] #neu
positive = stock_data.iloc[start:total_data,4] #pos
open_price = stock_data.iloc[start:total_data,5] #open
high = stock_data.iloc[start:total_data,6] #high
low = stock_data.iloc[start:total_data,7] #low
volume = stock_data.iloc[start:total_data,8] #volume

# printing close price
print("Close Price:")
close_price


# In[30]:


close_price_shifted = close_price.shift(-1) 

# shifting next day compound
compound_shifted = compound.shift(-1) 

# concatenating the captured training data into a dataframe
data = pd.concat([close_price, close_price_shifted, compound, compound_shifted, volume, open_price, high, low], axis=1)

# setting column names of the revised stock data
data.columns = ['close_price', 'close_price_shifted', 'compound', 'compound_shifted','volume', 'open_price', 'high', 'low']

# dropping nulls
data = data.dropna()    
data


# In[31]:



# setting the target variable as the shifted close_price
y = data['close_price_shifted']
y


# In[32]:


cols = ['close_price', 'compound', 'compound_shifted', 'volume', 'open_price', 'high', 'low']
x = data[cols]
x


# In[33]:


scaler_x = preprocessing.MinMaxScaler (feature_range=(-1, 1))
x = np.array(x).reshape((len(x) ,len(cols)))
x = scaler_x.fit_transform(x)

# scaling the target variable
scaler_y = preprocessing.MinMaxScaler (feature_range=(-1, 1))
y = np.array (y).reshape ((len( y), 1))
y = scaler_y.fit_transform (y)

# displaying the scaled feature dataset and the target variable
x, y


# In[34]:


X_train = x[0 : train_end,]
X_test = x[train_end+1 : len(x),]    
y_train = y[0 : train_end] 
y_test = y[train_end+1 : len(y)]  

# printing the shape of the train and the test datasets
print( X_train.shape, 'and y:', y_train.shape)
print( X_test.shape, 'and y:', y_test.shape)


# In[35]:


X_train = X_train.reshape (X_train.shape + (1,)) 
X_test = X_test.reshape(X_test.shape + (1,))

print( X_train.shape)
print( X_test.shape)


# In[36]:



# setting the seed to achieve consistent and less random predictions at each execution
np.random.seed(2016)

# setting the model architecture
model=Sequential()
model.add(LSTM(110,return_sequences=True,activation='tanh',input_shape=(len(cols),1)))
model.add(Dropout(0.1))
model.add(LSTM(110,return_sequences=True,activation='tanh'))
model.add(Dropout(0.1))
model.add(LSTM(110,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))

# printing the model summary
model.summary()


# In[37]:


# compiling the model
model.compile(loss='mse' , optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, epochs=12, batch_size=7, verbose=1)


# In[38]:


# saving the model as a json file
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights('model.h5')
print('Model is saved to the disk')


# In[39]:


# performing predictions
predictions = model.predict(X_test) 

# unscaling the predictions
predictions = scaler_y.inverse_transform(np.array(predictions).reshape((len(predictions), 1)))

# printing the predictions
print('Predictions:')
predictions[0:5]


# # Evaluating model

# In[40]:


train_loss = model.evaluate(X_train, y_train, batch_size = 1)
test_loss = model.evaluate(X_test, y_test, batch_size = 1)
print('Train Loss =', round(train_loss,4))
print('Test Loss =', round(test_loss,4))


# In[42]:


X_test = scaler_x.inverse_transform(np.array(X_test).reshape((len(X_test), len(cols))))
# unscaling datasets
y_train = scaler_y.inverse_transform(np.array(y_train).reshape((len(y_train), 1)))
y_test = scaler_y.inverse_transform(np.array(y_test).reshape((len(y_test), 1)))


# # Performing model on test data

# In[43]:


# plotting
plt.figure(figsize=(16,10))

# ploting on label="Training Close Price"
plt.plot(predictions, label="Predicted Close Price")
plt.plot([row[0] for row in y_test], label="Testing Close Price")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=2)
plt.show()


# In[ ]:




