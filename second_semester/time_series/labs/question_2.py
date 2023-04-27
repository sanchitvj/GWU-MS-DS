import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from toolbox import kpss_test, adf_test, plot_rolling_mean_var, non_seasonal_differencing, \
    gpac, auto_corr, acf_pacf_plot

import warnings
warnings.filterwarnings("ignore")
np.random.seed(6313)
lags = 50

df_orig = pd.read_csv("question2.csv")
print(df_orig.shape)

df = pd.DataFrame(columns=["value"])
# df_temp = pd.DataFrame(columns=["value"])
# df.loc[0, "value"] = np.float32(df_orig.columns[0])
# print(df)
# df_temp["value"] = df_orig['6.326500046330486571e-01']

ls = [6.326500046330486571e-01]
# ls.append(df_orig['6.326500046330486571e-01'].values.tolist())
ls[1:] = df_orig['6.326500046330486571e-01'].values.tolist()
print(ls[:5])
df["value"] = ls
# print(df)
# df = df_orig.copy()
date = pd.date_range(start='1981-01-01', periods=len(df), freq='D')
df.index = date
print(df.head().to_string())
#
df.plot()
plt.title("Time series dataset")
plt.xlabel("Time")
plt.ylabel("Data value")
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

res_kpss = kpss_test(df["value"])
res_adf = adf_test(df["value"])
plot_rolling_mean_var(df["value"])


stl = STL(df.value)
res = stl.fit()
fig = res.plot()
plt.show()

T, S, R = [res.trend, res.seasonal, res.resid]

strength_tr = max(0, 1 - (np.var(R)/np.var(R+T)))
strength_se = max(0, 1 - (np.var(R)/np.var(R+S)))

print("The strength of trend for this data set is: ", strength_tr)
print("The strength of seasonality for this data set is: ", strength_se)

print("Making dataset stationary by non seasonal differencing:")
y_diff_1 = non_seasonal_differencing(df.value, 1)
res_kpss = kpss_test(y_diff_1)
res_adf = adf_test(y_diff_1)
plot_rolling_mean_var(y_diff_1)

y = y_diff_1

j, k = 7, 7
ryt = auto_corr(y, lags, title=None, plot=False)
gpac_arr = gpac(ryt, j, k)
heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, k)])
# heatmap.add_patch(Rectangle((na - 1, nb), 1, j - nb, fill=False, edgecolor='green', lw=4))  # j line vertical
# heatmap.add_patch(
#     Rectangle((na, nb + 1), 1, k - 1 - na, fill=False, angle=270, edgecolor='blue', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, lags)


# arma_data, ryt, na, nb = sm_arma_process(15)
na, nb = 1, 0
# Estimate ARMA model coefficients from the generated data
model = sm.tsa.ARIMA(y, order=(na, 0, nb))  # ARIMA model with specified order
results = model.fit()  # method_kwargs={'warn_convergence': False})  # Fit the ARIMA model
# estimated_coefficients = results.arparams  # Get the estimated AR coefficients
print(results.summary())


y_train = y[:200]
y_test = y[200:]
model = sm.tsa.ARIMA(y_train, order=(0, 0, 0),  seasonal_order=(1, 1, 0, 3))
result = model.fit()
print(result.summary)

y_result_hat = result.predict(start=1, end=len(y_train)-1)
y_result_t = result.forecast(steps=len(y_test))
res_e = y_train[1:] - y_result_hat
fore_error = y_test - y_result_t


plt.plot(y_train, label='Train')
plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test, label='Test')
plt.plot(range(1, len(y_train)), y_result_hat, label='One-Step Predictions')
plt.legend()
plt.title('Train, Test, and One-Step Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()


print(f'forecast errors is = ', np.var(fore_error))
print(f'Forecast errors variance: {np.var(fore_error):.3f}')
print(f'Residual errors vs. forecast errors: {np.var(res_e)/np.var(fore_error):.3f}')
from scipy.stats import chi2

print('variance of residuals errors is = ', np.var(res_e))

re = auto_corr(res_e, lags, title=None, plot=False)
# acf_pacf_plot(res_e, lags)
Q = len(y_train) * np.sum(np.square(re[lags:]))
DOF = lags - na - nb
alfa = 0.01
chi_critical = chi2.ppf(1-alfa, DOF)
if Q < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")


lbvalue = sm.stats.acorr_ljungbox(res_e[1:100], model_df=7, boxpierce=True)
print(lbvalue)

# y_train = y[:950]
# y_test = y[950:]
# model = sm.tsa.ARIMA(y_train, order=(0, 0, 0), seasonal_order=(1, 1, 0, 3))
# model_fit = model.fit()
# pred_50 = model_fit.forecast(steps=50)
# print(f'forecast errors is = {np.var(pred_50 - y_test)}')
# plt.plot(range(len(y_test)), y_test, label='Test')
# plt.plot(range(len(y_test)), pred_50, label='50-Step Predictions')
# plt.legend()
# plt.title('Test vs 50-Step Predictions')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.show()

# corr_yt = np.correlate()

corr_coef = np.corrcoef(y_result_hat, y_result_t[:199])[0, 1]
print("Correlation Coefficient: ", corr_coef)
#
plt.scatter(range(len(y_result_hat)), y_result_hat)
plt.scatter(range(len(y_result_hat)), y[:199])
plt.title("Scatterplot of y and y_hat")
plt.xlabel("y")
plt.ylabel("y_hat")
plt.show()
