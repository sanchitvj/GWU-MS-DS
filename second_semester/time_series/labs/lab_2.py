import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import auto_corr
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()
np.random.seed(6313)

x = np.random.normal(0, 1, 1000)
date = pd.date_range(start='1/1/2000',
                     end='12/31/2000',
                     periods=len(x))

df = pd.DataFrame(x, columns=['white_noise'])
df.index = date

# print(df.head())

df.plot()
plt.show()

plt.hist(df.white_noise, bins=20)
plt.show()

print(f'Sampled Mean: {np.mean(df.white_noise):0.4f}')
print(f'Sampled Standard Deviation: {np.std(df.white_noise):0.4f}')

y = [3, 9, 27, 81, 243]
lag = 4

ryt = auto_corr(y, lag)
print(ryt)

ryt = auto_corr(df.white_noise, lag=20, title="White Noise")
print(ryt)

