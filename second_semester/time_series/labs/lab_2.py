import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import auto_corr, plt_subplot
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()
np.random.seed(6313)
colors = ['chartreuse', 'olive', 'salmon', 'teal', 'plum', 'lavender', 'navy']

x = np.random.normal(0, 1, 1000)
date = pd.date_range(start='1/1/2000',
                     end='12/31/2000',
                     periods=len(x))

df = pd.DataFrame(x, columns=['white_noise'])
df.index = date

df.plot(color=random.choice(colors))
plt.show()

plt.hist(df.white_noise, bins=20, color=random.choice(colors))
plt.show()

print(f'Sampled Mean: {np.mean(df.white_noise):0.4f}')
print(f'Sampled Standard Deviation: {np.std(df.white_noise):0.4f}')

y = [3, 9, 27, 81, 243]
lag = 4

ryt = auto_corr(y, lag)
print(ryt)

ryt = auto_corr(df.white_noise, lag=20, title="White Noise")
print(ryt)

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = '2000-01-01'
end_date = '2023-02-15'


df_ = pd.DataFrame()
for x in stocks:
    df = data.get_data_yahoo(x, start=start_date, end=end_date)
    df_[f"{x}"] = df.Close

# print(df_.tail().to_string())

subplot_data = []
subplot_title = []
for i in range(len(stocks)):
    subplot_data.append(df_[f"{df_.columns[i]}"])
    subplot_title.append(df_.columns[i])

plt_subplot(subplot_data, "Closing value of stocks", subplot_title, 3, 2, "Close price", "Year")

subplot_data_acf = []
# subplot_title_acf = []
for i in range(len(stocks)):
    df_new = pd.DataFrame(index=df_.index)
    df_new["close"] = df_[f"{df_.columns[i]}"]
    df_new = df_new.dropna()
    # print(df_new.head().to_string())
    subplot_data_acf.append(auto_corr(df_new["close"], lag=50, plot=False))

plt_subplot(subplot_data_acf, "ACF of closing value", subplot_title, 3, 2, "Magnitude", "Lag", acf=True, lag=50)


