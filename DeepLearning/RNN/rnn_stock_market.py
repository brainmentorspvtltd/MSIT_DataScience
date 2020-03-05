import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('SP500_train.csv')
dataset_test = pd.read_csv('SP500_test.csv')

trainigset = dataset_train.iloc[:,5:6].values
testset = dataset_test.iloc[:,5:6].values

minmax = MinMaxScaler()
scaled_trainset = minmax.fit_transform(trainigset)

X_train = []
y_train = []

for i in range(40,trainigset.shape[0]):
    X_train.append(scaled_trainset[i - 40 : i, 0])
    y_train.append(scaled_trainset[i,0])

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

#num_of_samples,num_of_features,1
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1))


# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=100,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50,))
model.add(Dropout(0.3))
model.add(Dense(units=1))

model.compile(optimizer='adam',
              loss='mean_squared_error')
model.fit(X_train,y_train,epochs=100,batch_size=32)

dataset_total = pd.concat((dataset_train['adj_close'],
                           dataset_test['adj_close']),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 40:].values
inputs = inputs.reshape(-1,1)

scaled_testset = minmax.fit_transform(inputs)
X_test = []
for i in range(40,len(testset) + 40):
    X_test.append(scaled_testset[i-40:i,0])

X_test = np.asarray(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

predictions = model.predict(X_test)

predictions = minmax.inverse_transform(predictions)

plt.plot(testset,color='b',label='Actual Prices')
plt.plot(predictions,color='green',label='LSTM Predictions')
plt.legend()
plt.show()





