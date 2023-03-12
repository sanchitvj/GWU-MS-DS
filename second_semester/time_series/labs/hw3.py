
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from toolbox import moving_average, plt_subplot, subplotting, adf_test
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("daily-min-temperatures.csv", index_col='Date')  # , parse_dates=['Date'])
date = pd.date_range(start='1981-01-01', periods=len(df), freq='D')
df.index = date

print(df.head().to_string())
# plt.figure(figsize=(12, 10))
# df.plot()
# plt.grid()
# plt.tight_layout()
# plt.show()

cycle_trend, _ = moving_average(df.Temp[:20])
print(cycle_trend)

data = []
data_orig = [df.Temp[:50], df.Temp[:50], df.Temp[:50], df.Temp[:50]]
detrended = []
title = ["3-MA", "5-MA", "7-MA", "9-MA"]
for i in range(4):
    cycle_trend, detrend = moving_average(df.Temp[:50])
    data.append(cycle_trend), detrended.append(detrend)

subplotting(df.index[:50], data, detrended, data_orig, "Cycle Trends", title, 2, 2, "Temperature", "Time")

data2 = []
detrended2 = []
title = ["2x4-MA", "2x6-MA", "2x8-MA", "2x10-MA"]
for i in range(4):
    cycle_trend, detrend = moving_average(df.Temp[:50])
    data2.append(cycle_trend), detrended2.append(detrend)

subplotting(df.index[:50], data2, detrended2, data_orig, "Cycle Trends", title, 2, 2, "Temperature", "Time")

cycle_trend, detrend = moving_average(df.Temp)
print("ADF test for original data: ")
adf_test(df.Temp)
print("ADF test for detrended data: ")
# print([x for x )
adf_test(detrend[1:len(detrend)-1])

temp = pd.Series(df.Temp.values, index=date, name="Daily Temperature")

stl = STL(temp, period=365)
res = stl.fit()
fig = res.plot()
plt.show()

T, S, R = [res.trend, res.seasonal, res.resid]

plt.plot(df.Temp, label="Original")
plt.plot((T+R), label="Seasonally adjusted")
plt.title("Original and Seasonally adjusted data")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()

strength_tr = max(0, 1 - (np.var(R)/np.var(R+T)))
strength_se = max(0, 1 - (np.var(R)/np.var(R+S)))

print("The strength of trend for this data set is: ", strength_tr)
print("The strength of seasonality for this data set is: ", strength_se)

# t = [x for x in range(50)]
# plt.plot(df.index, df.Temp, label="Original")
plt.plot(df.index, S, label="Seasonality")
plt.plot(df.index, T, label="Trend")
plt.plot(df.index, R, label="Residual")
plt.legend()
plt.tight_layout()
plt.xlabel("Time")
plt.ylabel("Temparature")
plt.show()

