# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
**.** The given problem is to predict the google stock price based on time.
**.** For this we are provided with a dataset which contains features like
**.** Date, Opening Price, Highest Price, Lowest Price, Closing Price, Adjusted Closing Price, Volume
**.** Based on the given features, develop a RNN model to predict, the price of stocks in futur
## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

1.Read the csv file and create the Data frame using pandas.
2.Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.
3.Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train.
4.Create a model with the desired number of neurons and one output neuron.
5.Follow the same steps to create the Test data. But make sure you combine the training data with the test data.
6.Make Predictions and plot the graph with the Actual and Predicted values.

## Program
Developed By: A.Sai Bandhavi.
Reg No: 212221240006.
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

df_train = pd.read_csv('trainset.csv')
df_train.head(60)

train_set = df_train.iloc[:,1:2].values
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)


X_train

X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train1

model = Sequential([layers.SimpleRNN(50,input_shape=(60,1)),
                    layers.Dense(1)
                    ])

model.compile(optimizer='Adam', loss='mae')

model.fit(X_train1,y_train,epochs=100,batch_size=32)

df_test=pd.read_csv("testset.csv")
test_set = df_test.iloc[:,1:2].values

dataset_total = pd.concat((df_train['Open'],df_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```

## OUTPUT

### True Stock Price, Predicted Stock Price vs time
![Screenshot 2023-09-27 103039](https://github.com/Saibandhavi75/rnn-stock-price-prediction/assets/94208895/2ad17c5a-1ead-4cb3-98d3-f3ede8c41054)



### Mean Square Error
![Screenshot 2023-09-27 103024](https://github.com/Saibandhavi75/rnn-stock-price-prediction/assets/94208895/e88353f5-243e-422e-a0f3-4d3c6c55f9f8)



## RESULT
A Recurrent Neural Network model for stock price prediction is developed.
