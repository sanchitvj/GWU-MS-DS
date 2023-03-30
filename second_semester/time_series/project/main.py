import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from toolbox import auto_corr, adf_test, kpss_test, cal_rolling_mean_var, \
    non_seasonal_differencing

df_orig = pd.read_csv("weather_madrid_2019-2022 2.csv", parse_dates=["time"])

# print(df_orig.head().to_string())
df_orig = df_orig.drop(['Unnamed: 0'], axis=1)
# print(df_orig.isnull().sum())
# print(df_orig.describe().to_string())
# print(df_orig.barometric_pressure.value_counts())

df = df_orig.copy()
df["time"] = df["time"].dt.tz_localize(None)
df.index = df.time
# print(df.head().to_string())

df.temperature.plot()
plt.ylabel("Temp")
plt.show()

mean, var = cal_rolling_mean_var(df.temperature)
plt.subplot(2, 1, 1)
plt.plot(mean, 'b', label="Varying mean")
plt.title('Rolling Mean - temperature')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(var, 'b', label="Varying variance")
plt.title('Rolling Variance - temperature')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

ryt = auto_corr(df.temperature, 100, "ACF of temperature")

adf_test(df.temperature)
kpss_test(df.temperature)

df["temperature_1st_order"] = non_seasonal_differencing(df.temperature, 1)

mean, var = cal_rolling_mean_var(df.temperature_1st_order)
plt.subplot(2, 1, 1)
plt.plot(mean, 'b', label="Varying mean")
plt.title('Rolling Mean - temperature')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(var, 'b', label="Varying variance")
plt.title('Rolling Variance - temperature')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

adf_test(df.temperature_1st_order)
kpss_test(df.temperature_1st_order)
ryt = auto_corr(df.temperature_1st_order, 100, "ACF of temperature")

# temp = pd.Series(df.temperature.values, index=time, name="Daily Temperature")

stl = STL(df.temperature.values, period=365)
res = stl.fit()
fig = res.plot()
plt.show()

T, S, R = [res.trend, res.seasonal, res.resid]

# plt.plot(df.temperature, label="Original")
# plt.plot((T+R), label="Seasonally adjusted")
# plt.title("Original and Seasonally adjusted data")
# plt.xlabel("Time")
# plt.ylabel("Temperature")
# plt.legend()
# plt.show()

strength_tr = max(0, 1 - (np.var(R) / np.var(R + T)))
strength_se = max(0, 1 - (np.var(R) / np.var(R + S)))

print("The strength of trend for this data set is: ", strength_tr)
print("The strength of seasonality for this data set is: ", strength_se)

gc.collect()
