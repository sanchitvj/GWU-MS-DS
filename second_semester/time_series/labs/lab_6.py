import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal
import seaborn as sns
from toolbox import ar_ma_dlsim, acf_pacf_plot, adf_test, gpac, auto_corr, \
    seasonal_differencing, plot_rolling_mean_var
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings("ignore")
np.random.seed(6313)

# N = int(input("Enter number of samples (N): "))
# mean = float(input("Enter the mean of white noise: "))
# var = float(input("Enter the variance of white noise: "))
# na = int(input("Enter AR order: "))
# nb = int(input("Enter MA order: "))
#
# ar_coeffs, ma_coeffs = [], []
# print("Enter the coefficients of AR (separated by space):")
# ar_coeffs = [float(x) for x in input().split()]
#
# print("Enter the coefficients of MA (separated by space):")
# ma_coeffs = [float(x) for x in input().split()]
#
# ar_params = np.array(ar_coeffs)
# ma_params = np.array(ma_coeffs)
lag = 20
mean, var, N = 0, 1, 1000
ar_params, ma_params = [1, 0, 0, -1.5, 0, 0, 0.5], [1, 0, 0, 0, 0, 0, 0]
seasonal_period = 3
na, nb = len(ar_params) - 1, 0
# e = np.random.normal(mean, var, N)
# y = ar_ma_dlsim(e, ma_params, ar_params, 'ar')

e = np.random.normal(mean, np.sqrt(var), size=N)
system = (ma_params, ar_params, 1)
tout, y_new = signal.dlsim(system, e)

y = np.ndarray.flatten(y_new)
formatted_data = [float("{:.2f}".format(num)) for num in y[:15]]
print(f"Generate data{na,nb}", formatted_data)

ar_params = np.array(ar_params)
ma_params = np.array(ma_params)
ar_params, ma_params = np.r_[ar_params], np.r_[ma_params]
arma_process = sm.tsa.ArmaProcess(ar_params, ma_params)
# arma_process = sm.tsa.ARIMA(ar_params, ma_params, order=(na, 0, nb), seasonal_order=(0, 0, 0, 3))
# white_noise = np.sqrt(variance) * np.random.randn(N) + mean
# mean_y = mean * ((1 + np.sum(ma_params)) / (1 + np.sum(ar_params)))
# arma_data = arma_process.generate_sample(nsample=N, scale=np.sqrt(var), distrvs=None, burnin=0,
#                                          axis=0) + mean_y

# y = arma_data
# print(y)
acf_pacf_plot(y, lag)

result = adf_test(y)

plot_rolling_mean_var(y)

print("Is this process stationary: ", arma_process.isstationary)

y_list = []
j, k = 9, 9
if result[1] < 0.05:
    # ryt = arma_process.acf(lags=lag)
    ryt = auto_corr(y, lag, title=None, plot=False)
    gpac_arr = gpac(ryt, j, k)
    heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, k)])
    heatmap.add_patch(Rectangle((na - 1, nb), 1, j - nb, fill=False, edgecolor='green', lw=4))  # j line vertical
    heatmap.add_patch(
        Rectangle((na, nb + 1), 1, k - 1 - na, fill=False, angle=270, edgecolor='blue', lw=4))  # k line horizontal
    plt.title("GPAC Table")
    plt.show()
    pass

else:
    y_diff = seasonal_differencing(y, seasonal_period)
    # print(y_diff.shape)
    acf_pacf_plot(y_diff, lag)
    result = adf_test(y_diff)
    # print(y_diff)
    plot_rolling_mean_var(np.array(y_diff))
    if result[1] < 0.05:
        # y = y_diff
        y_list.append(y_diff)
        print(f"Dataset is stationary after {seasonal_period} order differencing.")
        ryt = acf(y_diff, nlags=lag)
        gpac_arr = gpac(ryt, j, k)
        heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, k)])
        heatmap.add_patch(
            Rectangle((na - 1, nb), 1, j - nb, fill=False, edgecolor='green', lw=4))  # j line vertical
        heatmap.add_patch(Rectangle((na, nb + 1), 1, k - 1 - na, fill=False, angle=270, edgecolor='blue',
                                    lw=4))  # k line horizontal
        plt.title("GPAC Table")
        plt.show()


plt.plot(y[:500], label="Raw Dataset")
if len(y_list) != 0:
    plt.plot(y_diff.reshape(-1, 1)[:500], color="orange", label="Differenced Dataset")
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Data")
plt.title("Sample of dataset")
plt.show()


y_train = y[:100]
y_test = y[100:]
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

print("\n-----Ans-12.------\n")

print(f'forecast errors is = ', np.var(fore_error))
print(f'Forecast errors variance: {np.var(fore_error):.3f}')
print(f'Residual errors vs. forecast errors: {np.var(res_e)/np.var(fore_error):.3f}')
from scipy.stats import chi2
lags = 20

print('variance of residuals errors is = ', np.var(res_e))

lags = 20
re = auto_corr(res_e, lags, title="ACF of forecast function", plot=True)
acf_pacf_plot(res_e, lags)
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

y_train = y[:950]
y_test = y[950:]
model = sm.tsa.ARIMA(y_train, order=(0, 0, 0), seasonal_order=(1, 1, 0, 3))
model_fit = model.fit()
pred_50 = model_fit.forecast(steps=50)
print(f'forecast errors is = {np.var(pred_50 - y_test)}')
plt.plot(range(len(y_test)), y_test, label='Test')
plt.plot(range(len(y_test)), pred_50, label='50-Step Predictions')
plt.legend()
plt.title('Test vs 50-Step Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()


