import pandas as pd
import numpy as np
import matplotlib.pyplot as olt
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets

df = pd.read_csv("AirPassengers.csv", index_col='Month', parse_dates=True)

print(df.head())

y = df["#Passengers"]
lags = 40
ACF_PACF_Plot(y, lags)

yt, yf = train_test_split(y, shuffle=False, test_size=0.2)


holtt = ets.ExponentialSmoothing(yt, trend=None, damped=False, seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values, np.ndarray.flatten(holtf.values))).mean()
print(MSE)

