from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import math
# import tensorflow as tf
from tensorflow.keras import Sequential
# from keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense, LSTM, Dropout #,CuDNNLSTM
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# import tensorflow as tf
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.debugging.set_log_device_placement(True)
CUDA_VISIBLE_DEVICES = '0,1'

df = data.get_data_yahoo('AAPL',start='2012-01-01',end='2023-04-19')
plt.figure(figsize=(16,8))
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['b'])
plt.plot(df['Close'])
plt.title(f'Clsoe price history of Apple')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USC($)')
plt.grid()
plt.show()

df.drop(columns=['Volume'], inplace= True)
data = df

scaler = StandardScaler()
scaler = scaler.fit(data)
scaled_data = scaler.transform(data)
df_close = df['Close'].values
dataset = data.values
training_data_len = math.ceil(len(df_close) *.8)
train_data = scaled_data[0:training_data_len,:]

x_train = []
y_train = []
n_past = len(dataset) - training_data_len

for i in range(n_past,len(train_data)):
    x_train.append(train_data[i-60:i,0:train_data.shape[1]-1])
    y_train.append(train_data[i,train_data.shape[1]-1])

    if i<=61:
        print(x_train)
        print(y_train)
        print()
# convert the x_train and y_train to numpy
x_train, y_train = np.array(x_train), np.array(y_train)
print(f'trainX shape == {x_train.shape}')
print(f'trainY shape == {y_train.shape}')


# Build LSTM Model

model = Sequential()
model.add(LSTM(64, return_sequences=True,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(50,return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.summary()
history = model.fit(x_train,y_train, batch_size=16, validation_split = .1, epochs=5, verbose=1)

plt.figure()
plt.plot(history.history['loss'], 'r',label='Training loss')
plt.plot(history.history['val_loss'], 'b',label='Validation loss')
plt.legend()
plt.show()


test_data = scaled_data[training_data_len-60 :,:  ]
x_test = []
y_test = dataset[training_data_len:,-1]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0:4])


# Convert the data to a numpy array
x_test = np.array(x_test)
predictions = model.predict(x_test)
forecast_copies = np.repeat(predictions, 5, axis=-1)
predictions = scaler.inverse_transform(forecast_copies)[:,0]

train = data.iloc[:training_data_len]
valid = data.iloc[training_data_len:]
valid['Predictions'] = predictions

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.set_title("Apple stock (closing price) prediction using LSTM network",fontsize=18)
ax.set_xlabel("Date",fontsize=18)
ax.set_ylabel("Adj Close price USD($)", fontsize=18)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['r','b','c'])
ax.plot(train["Adj Close"],'blue')
ax.plot(valid["Adj Close"],'red')
ax.plot(valid["Predictions"],'black')
ax.legend(["Train", "Val", "Predictions"], loc = "lower right", fontsize=18)
ax.grid()
plt.show()

# from lifelines import KaplanMeierFitter
# import matplotlib.pyplot as plt
#
# ax = plt.subplot(111)
# durations = [5,6,6,2.5,4,4]
# event = [1,0,0,1,1,1]
# kmf = KaplanMeierFitter()
# kmf.fit(durations,event, label = 'Number of users stay on a website')
# kmf.plot_survival_function(ax = ax)
# plt.show()