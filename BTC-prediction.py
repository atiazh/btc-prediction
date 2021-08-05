#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:15:29 2021

@author: andriitiazhkyi
"""

import pandas as pd
from time import sleep
#from binance.client import Client
import datetime
from time import sleep
from datetime import datetime
from time import mktime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, LeakyReLU, Dropout, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Hyper parmeters
SEQ_LEN = 48                               # last 60 minutes
FUTURE_PERIOD_PREDICT = 12                  # predict movement in 3 minutes
RATIO_TO_PREDICT = "BTC-USD"
currencies = ["BTCUSDT", "BCHABCUSDT", "ETHUSDT", "EOSUSDT", "LTCUSDT"]
OPEN_TIME_IDX = 0
OPEN_IDX = 1
HIGH_IDX = 2
LOW_IDX = 3
CLOSE_IDX = 4
VOLUME_IDX = 5
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.0003
DECAY = 1e-5
LOSS = 'sparse_categorical_crossentropy'
INPUT_DIM = 14 


#api_key = ''
#api_secret = ''
client = Client(api_key, api_secret)

# STEP 1: collect data
data = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 Jan, 2020")
columns = ['time','open','high','low','close','volume','close_time','quote_asset_volume','num_trades','ignore','ignore_2','ignore_3']
df = pd.DataFrame(data, columns = columns)
df = df.sort_values('time',ascending=False)
df.drop('quote_asset_volume', axis=1, inplace=True)
df.drop('num_trades', axis=1, inplace=True)
df.drop('close_time', axis=1, inplace=True)
df.drop('ignore', axis=1, inplace=True)
df.drop('ignore_2', axis=1, inplace=True)
df.drop('ignore_3', axis=1, inplace=True)
df = df.reset_index()
df.drop('index', axis=1, inplace=True)
df.info()

# OR upload the data 
df = pd.read_csv('BTCUSDT_1min_2020.csv')
# convert time to normal time
df['time'] = df['time']/1000
df['time'] = df['time'].astype(int)
df['date'] = [datetime.fromtimestamp(x) for x in df['time']]

# STEP 2: preporocess
# take 50000 rows = 1 month
df_try = df.iloc[:50000]
# sort values
df_try = df_try.sort_values('date')
df_try = df_try.reset_index()

# normalize prices
min_max_scaler = MinMaxScaler()
price = df_try[['close']]
norm_data = min_max_scaler.fit_transform(price.values)

# split data 
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)
# past history - how many samlples we use to 
past_history = 5
future_target = 0

TRAIN_SPLIT = int(len(norm_data) * 0.8)



x_train, y_train = univariate_data(norm_data,
                                   0,
                                   TRAIN_SPLIT,
                                   past_history,
                                   future_target)

x_test, y_test = univariate_data(norm_data,
                                 TRAIN_SPLIT,
                                 None,
                                 past_history,
                                 future_target)


y = df_try.pop('close')
x = df_try.copy(deep=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# make a model
num_units = 64
learning_rate = 0.0001
activation_function = 'sigmoid'
adam = Adam(lr=learning_rate)
loss_function = 'mse'
batch_size = 5
num_epochs = 50

# Initialize the RNN
model = Sequential()
model.add(LSTM(units = num_units, activation=activation_function, input_shape=(None, 1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.1))
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer=adam, loss=loss_function)

def build_model(loss, opt):
    model = Sequential()
    # model.add(Dense(64, activation='tanh', input_shape=(SEQ_LEN, INPUT_DIM)))
    # model.add(BatchNormalization())

    model.add(LSTM(256, input_shape=(SEQ_LEN, INPUT_DIM), return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())

    model.add(LSTM(256))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.1))

    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=['accuracy']
    )
    return model

opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE) #, decay=DECAY)
model = build_model(LOSS, opt)
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle=False)

# plot loss function
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.show()

# plot predictions
original = min_max_scaler.inverse_transform(y_test)
originals = pd.DataFrame(original)
prediction = min_max_scaler.inverse_transform(model.predict(x_test))
predictions = pd.DataFrame(prediction)
ax = sns.lineplot(x=original.index, y=original[0], label="Test Data", color='royalblue')
ax = sns.lineplot(x=predictions.index, y=predictions[0], label="Prediction", color='tomato')
ax.set_title('Bitcoin price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)

model.evaluate(x_test, y_test)


### accuracy metrics
# mean_absolute_percentage_error (MAPE)
def mean_absolute_percentage_error(y_actual, y_pred):
  y_true, y_pred = np.array(y_actual), np.array(y_pred)
  accuracy = (np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100)
  return print('Accuracy is:', accuracy)

mean_absolute_percentage_error(original, prediction)