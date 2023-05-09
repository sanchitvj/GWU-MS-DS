import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from toolbox import auto_corr, adf_test, kpss_test, cal_rolling_mean_var, \
    non_seasonal_differencing, average_method, naive_method, drift_method, \
    simple_expo_smoothing_method, box_pierce_test_q_value, acf_pacf_plot, \
    seasonal_differencing, forecasting, forecast_plot_test, backward_regression, \
    gpac, arima_model

seed_num = 6313
np.random.seed(seed_num)

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
# plt.show()

# mean, var = cal_rolling_mean_var(df.temperature)
# plt.subplot(2, 1, 1)
# plt.plot(mean, 'b', label="Varying mean")
# plt.title('Rolling Mean - temperature')
# plt.xlabel('Samples')
# plt.ylabel('Magnitude')
#
# plt.subplot(2, 1, 2)
# plt.plot(var, 'b', label="Varying variance")
# plt.title('Rolling Variance - temperature')
# plt.xlabel('Samples')
# plt.ylabel('Magnitude')
# plt.legend()
#
# plt.subplots_adjust(bottom=0.15)
# plt.tight_layout(h_pad=2.2, w_pad=2)
# plt.show()

# ryt = auto_corr(df.temperature[:500], 50, "ACF of temperature")

# adf_test(df.temperature)
# kpss_test(df.temperature)

# df["temperature_1st_order"] = non_seasonal_differencing(df.temperature, 1)
#
# print("Stats after 1st order non-seasonal differencing:")
# mean, var = cal_rolling_mean_var(df.temperature_1st_order)
# plt.subplot(2, 1, 1)
# plt.plot(mean, 'b', label="Varying mean")
# plt.title('Rolling Mean - temperature')
# plt.xlabel('Samples')
# plt.ylabel('Magnitude')
#
# plt.subplot(2, 1, 2)
# plt.plot(var, 'b', label="Varying variance")
# plt.title('Rolling Variance - temperature')
# plt.xlabel('Samples')
# plt.ylabel('Magnitude')
# plt.legend()
#
# plt.subplots_adjust(bottom=0.15)
# plt.tight_layout(h_pad=2.2, w_pad=2)
# plt.show()

# adf_test(df.temperature_1st_order)
# kpss_test(df.temperature_1st_order)
# ryt = auto_corr(df.temperature_1st_order[:500], 50, "ACF of temperature")
y = seasonal_differencing(df.temperature, 24)
y = non_seasonal_differencing(y, 1)
acf_pacf_plot(y, 50)
df["stationary_data"] = y

stl = STL(df.temperature.values, period=24)
res = stl.fit()
fig = res.plot()
# plt.show()

T, S, R = [res.trend, res.seasonal, res.resid]
strength_tr = max(0, 1 - (np.var(R) / np.var(R + T)))
strength_se = max(0, 1 - (np.var(R) / np.var(R + S)))
print("The strength of trend for this data set is: ", strength_tr)
print("The strength of seasonality for this data set is: ", strength_se)

stl = STL(df.stationary_data.values, period=24)
res = stl.fit()
fig = res.plot()
# plt.show()

T, S, R = [res.trend, res.seasonal, res.resid]
strength_tr = max(0, 1 - (np.var(R) / np.var(R + T)))
strength_se = max(0, 1 - (np.var(R) / np.var(R + S)))
print("The strength of trend for this data set is: ", strength_tr)
print("The strength of seasonality for this data set is: ", strength_se)

n_train = int(len(df) * 0.8)
train = df.stationary_data[:n_train]
test = df.stationary_data[n_train:]
model = ExponentialSmoothing(train, seasonal_periods=24)
fit = model.fit()
preds = fit.forecast(len(test))
plt.plot(df.index[:n_train], train, label='Train')
plt.plot(df.index[n_train:], test, label='Test')
plt.plot(df.index[n_train:], preds, label='Predictions')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Temperature")
# plt.show()

data = df.drop(["temperature", "time"], axis=1)
sns.heatmap(data.corr(), annot=True)
plt.tight_layout()
# plt.show()

X = df.drop(["temperature", "time", "stationary_data"], axis=1)
y = df["stationary_data"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=seed_num)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

x_train = pd.DataFrame(x_train_s, columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(x_test_s, columns=x_test.columns, index=x_test.index)

# feats = backward_regression(x_train, x_train, y_train)
# print("Selected features: ", feats.values)
# print("Removed features: ", [x for x in x_train.columns.values if x not in feats.values])
# removed_features = []
# model_results = []
# model = []
# # Backward Selection
# for subject in subjects:
#     removed_features = []
#     model_results = []
#     model = []
#     X_train_ = sm.add_constant(x_train.copy())
#     for step in range(len(predictors) - 1):
#         if step != 0:
#             X_train_ = X_train_.drop(columns=removed_features[step])
#
#         model[step] = sm.OLS(y_train, X_train_).fit()
#         print(model[subject][step].summary())
#         model_results[subject][step] = {'AIC': model[subject][step].aic,
#                                         'BIC': model[subject][step].bic,
#                                         'RMSE': model[subject][step].mse_model ** 0.5,
#                                         'Adj Rsq': model[subject][step].rsquared_adj,
#                                         'Rsq': model[subject][step].rsquared}
#         if model[subject][step].pvalues.sort_values(ascending=False).iloc[0] > 0.001:
#             removed_features[subject][step + 1] = model[subject][step].pvalues.sort_values(ascending=False).index[0]
#         else:
#             break
#     model_results[subject] = pd.DataFrame.from_dict(model_results[subject],
#                                                     orient='index').sort_values('BIC',
#                                                                                 ascending=False)

H = np.dot(x_train.T, x_train)
s, d, v = np.linalg.svd(H, full_matrices=False)
print("Singular values: ", d)

condition_num = np.linalg.cond(x_train)
print("Condition number: ", condition_num)

vif = pd.DataFrame()
vif["Feature"] = x_train.columns
vif["VIF"] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
print(vif)

# t = [i for i in range(len(df))]
# n_train = len(x_train)
# yt_hat, e_a, e2 = forecasting("average", df.stationary_data, n_train=n_train)
# res_avg = forecast_plot_test(df.stationary_data, t, e_a, e2, yt_hat, "Average", n_train, lag=50)
#
# yt_hat, e_n, e2 = forecasting("naive", df.stationary_data, n_train=n_train)
# res_naive = forecast_plot_test(df.stationary_data, t, e_n, e2, yt_hat, "Naive", n_train, lag=50)
#
# yt_hat, e_d, e2 = forecasting("drift", df.stationary_data, n_train=n_train)
# res_drift = forecast_plot_test(df.stationary_data, t, e_d, e2, yt_hat, "Drift", n_train, lag=50)
#
# yt_hat, e_s, e2 = forecasting("ses", df.stationary_data, n_train=n_train, alpha=0.25)
# res_ses = forecast_plot_test(df.stationary_data, t, e_s, e2, yt_hat, "SES", n_train, alpha=0.25, lag=50)
#
# df_res = pd.DataFrame(columns=["Mean Prediction Error", "MSE Training",
#                                "MSE Testing", "Variance Training",
#                                "Variance Testing", "Q-value"])
# df_res.loc["Average", :] = res_avg
# df_res.loc["Naive", :] = res_naive
# df_res.loc["Drift", :] = res_drift
# df_res.loc["SES", :] = res_ses
# print(df_res.to_string())


X = sm.add_constant(x_train)
final_model = sm.OLS(y_train, X).fit()
print(final_model.summary())

# x_test = x_test
# x_test_s = scaler.transform(x_test)
X_test = sm.add_constant(x_test)
# print(X_test.shape)
y_pred = final_model.predict(X_test)

plt.plot(y_train.index, y_train, label='Train')
plt.plot(y_test.index, y_test.values, label='Test')
plt.plot(y_test.index, y_pred.values, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Price Forecasting')
plt.legend()
# plt.show()

print("t-test: ", final_model.tvalues)
print(final_model.pvalues)
print("f-test: ", final_model.fvalue)
print(final_model.f_pvalue)
print("AIC: ", final_model.aic)
print("BIC: ", final_model.bic)
forecast_rmse = np.mean(np.square(y_pred - y_test)) ** 0.5
print("RMSE value: ", forecast_rmse)
print("R-squared value:", final_model.rsquared)
print("Adjusted r-squared:", final_model.rsquared_adj)
acf_pacf_plot(final_model.resid, 50)
ryt = auto_corr(final_model.resid, 50, plot=False)
q_val = box_pierce_test_q_value(ryt, 50, len(ryt))
print("Q-value: ", q_val)
print("Mean of residuals: ", np.mean(final_model.resid))
print("Variance of residuals: ", np.var(final_model.resid))


#%%
j, k = 26, 26
ryt = auto_corr(y_train, 50, plot=False)
gpac_arr = gpac(ryt, j, k)
if abs(np.nanmin(gpac_arr)) <= abs(np.nanmax(gpac_arr)):
    vmax = abs(np.nanmax(gpac_arr))
    vmin = -vmax
else:
    vmax = abs(np.nanmin(gpac_arr))
    vmin = -vmax
plt.figure(figsize=(20, 15))
sns.heatmap(gpac_arr, annot=True, linewidths=0.5, xticklabels=[i for i in range(1, k)])
plt.title("GPAC Table")
plt.tight_layout()
plt.savefig("gpac.png")
plt.show()


#%%
# na, nb = 0, 1 -> According to ACF/PACF plot
# na, nb = 1, 0 -> According to GPAC Table

# lags = 50
# na, nb = 1, 1

# arima_model(0, 1, 50, y_train, y_test)
# arima_model(1, 0, 50, y_train, y_test)
arima_model(1, 1, 50, y_train, y_test)

#%%



