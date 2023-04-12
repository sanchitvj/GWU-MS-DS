import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from toolbox import auto_corr, adf_test, kpss_test, cal_rolling_mean_var, \
    non_seasonal_differencing, average_method, naive_method, drift_method, \
    simple_expo_smoothing_method, box_pierce_test_q_value

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


def forecasting(method, yt, n_train=None, alpha=0.5):
    yt_hat = []
    e = []
    e2 = []
    for i in range(len(yt)):
        if i == 0 and method != "ses":
            yt_hat.append(np.nan)
            e.append(np.nan)
            e2.append(np.nan)
        else:
            if method == "average":
                y_hat = average_method(yt, i, n_train)
            elif method == "naive":
                y_hat = naive_method(yt, i, n_train)
            elif method == "drift":
                y_hat = drift_method(yt, i, n_train)
            elif method == "ses":
                y_hat = simple_expo_smoothing_method(yt, yt_hat, i, n_train, alpha)
            else:
                y_hat = yt[i]
            yt_hat.append(round(y_hat, 3))
            e.append(round((yt[i] - y_hat), 3))
            e2.append(round((yt[i] - y_hat) ** 2, 3))

    return yt_hat, e, e2


def forecast_plot_test(yt, t, e, e2, cap_yt, method, n_train=26306, results=True, alpha=None, lag=5, tn=None):
    plt.figure(figsize=(10, 8))
    if method != "SES":
        plt.plot(t[:n_train], yt[:n_train], color='blue', label="training dataset")
        plt.plot(t[n_train:], yt[n_train:], color='orange', label="test dataset")
        plt.plot(t[n_train:], cap_yt[n_train:], color='green', label=f"{method} method h-step prediction")
        plt.title(f"{method} Method Forecast")
    else:
        # plt.plot(t, yt, color='blue', label="original")
        # plt.plot(t[:9], cap_yt[:9], color='orange', label=f"{method} 1-step prediction")
        # plt.plot(t[9:], cap_yt[9:], color='green', label=f"{method} h-step prediction")
        plt.plot(t[:n_train], yt[:n_train], color='blue', label="training dataset")
        plt.plot(t[n_train:], yt[n_train:], color='orange', label="test dataset")
        plt.plot(t[n_train:], cap_yt[n_train:], color='green', label=f"{method} method h-step prediction")
        plt.title(f"{method} Method Forecast with alpha = {alpha}")
    plt.ylabel("yt")
    plt.xlabel("time")
    plt.legend(loc='lower left')
    plt.show()

    mse_tr, mse_ts = round(np.nanmean(e2[2:n_train]), 3), round(np.nanmean(e2[n_train:]), 3)
    var_tr, var_ts = round(np.nanvar(e[2:n_train]), 3), round(np.nanvar(e[n_train:]), 3)
    q_value = box_pierce_test_q_value(e[2:n_train], lag, tn - 2)
    mean_e = round(np.mean(e[2:n_train]), 3)
    res = [mean_e, mse_tr, mse_ts, var_tr, var_ts, q_value]
    if results:
        print("\n")
        print(f"Results using {method} method forecast: ")
        # print("y_hat_t: ", cap_yt)
        # print("Error: ", e)
        # print("Error-squared: ", e2)
        print("MSE training: ", mse_tr)
        print("MSE testing: ", mse_ts)
        print("Variance training: ", var_tr)
        print("Variance testing: ", var_ts)
        print("Q-value: ", q_value)

    return res


t = [i for i in range(len(df))]
yt_hat, e_a, e2 = forecasting("average", df.temperature, n_train=26306)
res_avg = forecast_plot_test(df.temperature, t, e_a, e2, yt_hat, "Average", lag=50, tn=26306)

yt_hat, e_n, e2 = forecasting("naive", df.temperature, n_train=26306)
res_naive = forecast_plot_test(df.temperature, t, e_n, e2, yt_hat, "Naive", lag=50, tn=26306)

yt_hat, e_d, e2 = forecasting("drift", df.temperature, n_train=26306)
res_drift = forecast_plot_test(df.temperature, t, e_d, e2, yt_hat, "Drift", lag=50, tn=26306)

yt_hat, e_s, e2 = forecasting("ses", df.temperature, n_train=26306, alpha=0.25)
res_ses = forecast_plot_test(df.temperature, t, e_s, e2, yt_hat, "SES", alpha=0.25, lag=50, tn=26306)

df_res = pd.DataFrame(columns=["Mean Prediction Error", "MSE Training",
                               "MSE Testing", "Variance Training",
                               "Variance Testing", "Q-value"])
df_res.loc["Average", :] = res_avg
df_res.loc["Naive", :] = res_naive
df_res.loc["Drift", :] = res_drift
df_res.loc["SES", :] = res_ses
print(df_res.to_string())

gc.collect()
