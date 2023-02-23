import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from toolbox import auto_corr, cal_rolling_mean_var
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("daily-min-temperatures.csv", index_col='Date')  # , parse_dates=['Date'])
date = pd.date_range(start='1981-01-01', periods=len(df), freq='D')
df.index = date

plt.figure(figsize=(12, 10))
df.plot()
plt.grid()
plt.tight_layout()
# plt.show()

temp = pd.Series(df.Temp.values, index=date, name="Daily_Temperature")

stl = STL(temp)
res = stl.fit()
fig = res.plot()
plt.show()

T, S, R = [res.trend, res.seasonal, res.resid]

strength_tr = max(0, 1 - (np.var(R)/np.var(R+T)))
strength_se = max(0, 1 - (np.var(R)/np.var(R+S)))

print("Strength trend: ", strength_tr)
print("Strength seasonality: ", strength_se)

t = [x for x in range(50)]
plt.plot(t, df.Temp[:50], label="Original")
plt.plot(t, S[:50], label="Seasonality")
plt.plot(t, T[:50], label="Trend")
plt.plot(t, R[:50], label="Residual")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()


# ryt = auto_corr(df.Temp, lag=50, title="ACF")
#
# rm, rv = cal_rolling_mean_var(df, "Temp")
# plt.subplot(2, 1, 1)
# plt.plot(rm, 'b', color='cyan')
# plt.title('Rolling Mean')
# plt.xlabel('Samples')
# plt.ylabel('Magnitude')
#
# plt.subplot(2, 1, 2)
# plt.plot(rv, 'b', label="Varying variance", color='cyan')
# plt.title('Rolling Variance')
# plt.xlabel('Samples')
# plt.ylabel('Magnitude')
# plt.legend()
#
# plt.subplots_adjust(bottom=0.15)
# plt.tight_layout(h_pad=2.2, w_pad=2)
# plt.show()
