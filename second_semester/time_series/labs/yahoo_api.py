import pandas as pd
from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt
yf.pdr_override()

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM' 'YELP', 'MSFT']
start_date = '2000-01-01'
end_date = '2023-02-02'
df = data.get_data_yahoo(stocks[0], start=start_date, end=end_date)
# df2 = data.get_data_yahoo(stocks[1], start=start_date, end=end_date)
# df3 = data.get_data_yahoo(stocks[2], start=start_date, end=end_date)

# print(df.index)#tail().to_string())

col = df.columns
df[col[:-1]].plot()
plt.show()

# df_ = pd.DataFrame(index=df2.index)
# df_["apple"] = df1.Close
# df_["oracle"] = df2.Close
# df_["tesla"] = df3.Close
#
# print(df_.tail().to_string())

# plt.figure(figsize=(16,8))
# plt.subplot(3,1,1)
# plt.plot(df.Close, label="Cost")
# plt.title('Closing cost')
# plt.xlabel('Date')
# plt.ylabel('Cost')
# # plt.legend()
# plt.show()

# df.plot(subplots=True, figsize=(16,8), sharex=True, grid=True)
# plt.tight_layout()
# plt.xlabel('Year')
# plt.ylabel('Stock Price')
# plt.suptitle('Stock Price for Companies', fontsize=16)
# plt.show()