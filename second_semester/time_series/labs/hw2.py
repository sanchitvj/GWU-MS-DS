import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from toolbox import box_pierce_test_q_value, average_method, plt_subplot,\
    naive_method, drift_method, simple_expo_smoothing_method, auto_corr
import warnings
warnings.filterwarnings('ignore')

yt_1 = [112, 118, 132, 129, 121, 135, 148, 136, 119, 104, 118, 115, 126, 141]
t = [x + 1 for x in range(14)]
n_train = 9
alpha = 0.5
lag = 5

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


def forecast_plot_test(yt, t, e, e2, cap_yt, method, results=True, alpha=None, lag=5, tn=None):
    if method != "SES":
        plt.plot(t[:9], yt[:9], color='blue', label="training dataset")
        plt.plot(t[9:], yt[9:], color='orange', label="test dataset")
        plt.plot(t[9:], cap_yt[9:], color='green', label=f"{method} method h-step prediction")
        plt.title(f"{method} Method Forecast")
    else:
        # plt.plot(t, yt, color='blue', label="original")
        # plt.plot(t[:9], cap_yt[:9], color='orange', label=f"{method} 1-step prediction")
        # plt.plot(t[9:], cap_yt[9:], color='green', label=f"{method} h-step prediction")
        plt.plot(t[:9], yt[:9], color='blue', label="training dataset")
        plt.plot(t[9:], yt[9:], color='orange', label="test dataset")
        plt.plot(t[9:], cap_yt[9:], color='green', label=f"{method} method h-step prediction")
        plt.title(f"{method} Method Forecast with alpha = {alpha}")
    plt.ylabel("yt")
    plt.xlabel("time")
    plt.legend(loc='lower left')
    plt.show()

    mse_tr, mse_ts = round(np.nanmean(e2[2:9]), 3), round(np.nanmean(e2[9:]), 3)
    var_tr, var_ts = round(np.nanvar(e[2:9]), 3), round(np.nanvar(e[9:]), 3)
    q_value = box_pierce_test_q_value(e[2:9], lag, tn-2)
    mean_e = round(np.mean(e[2:9]), 3)
    res = [mean_e, mse_tr, mse_ts, var_tr, var_ts, q_value]
    if results:
        print("\n")
        print(f"Results using {method} method forecast: ")
        print("y_hat_t: ", cap_yt)
        print("Error: ", e)
        print("Error-squared: ", e2)
        print("MSE training: ", mse_tr)
        print("MSE testing: ", mse_ts)
        print("Variance training: ", var_tr)
        print("Variance testing: ", var_ts)
        print("Q-value: ", q_value)

    return res


yt_hat, e_a, e2 = forecasting("average", yt_1, n_train)
res_avg = forecast_plot_test(yt_1, t, e_a, e2, yt_hat, "Average", lag=5, tn=n_train)

yt_hat, e_n, e2 = forecasting("naive", yt_1, n_train)
res_naive = forecast_plot_test(yt_1, t, e_n, e2, yt_hat, "Naive", lag=5, tn=n_train)

yt_hat, e_d, e2 = forecasting("drift", yt_1, n_train)
res_drift = forecast_plot_test(yt_1, t, e_d, e2, yt_hat, "Drift", lag=5, tn=n_train)

yt_hat, e_s, e2 = forecasting("ses", yt_1, n_train, alpha)
res_ses = forecast_plot_test(yt_1, t, e_s, e2, yt_hat, "SES", alpha=alpha, lag=5, tn=n_train)

alphas = [0, 0.25, 0.75, 0.99]
data = []
plt.figure(figsize=(20,20))
fig, axes = plt.subplots(nrows=2, ncols=2)
for i, ax in enumerate(axes.flat):
    yt_hat, _, _ = forecasting("ses", yt_1, n_train, alphas[i])
    ax.plot(t[:9], yt_1[:9], color='blue', label="training dataset")
    ax.plot(t[9:], yt_1[9:], color='orange', label="test dataset")
    ax.plot(t[9:], yt_hat[9:], color='green', label=f"SES method h-step prediction")
    ax.set_title(f"alpha = {alphas[i]}")
    ax.set_ylabel("yt")
    ax.set_xlabel("time")
    tick_positions = [i for i in range(2, 15, 2)]
    tick_labels = [str(i) for i in range(2, 15, 2)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc='lower left', fontsize=5)

# fig.legend(loc='upper right')
plt.tight_layout()
# fig.tight_layout(rect=[0, 0, 0.75, 1])
plt.subplots_adjust(bottom=0.1, left=0.11, top=0.88)
fig.suptitle("SES method forecast with different alphas")
plt.show()

df = pd.DataFrame(columns=["Mean Prediction Error", "MSE Training",
                           "MSE Testing", "Variance Training",
                           "Variance Testing", "Q-value"])

df.loc["Average", :] = res_avg
df.loc["Naive", :] = res_naive
df.loc["Drift", :] = res_drift
df.loc["SES", :] = res_ses
print(df.to_string())

lag = 5  # 20
ryt_avg = auto_corr(e_a[2:9], lag, plot=False)
ryt_naive = auto_corr(e_n[2:9], lag, plot=False)
ryt_drift = auto_corr(e_d[2:9], lag, plot=False)
ryt_ses = auto_corr(e_s[2:9], lag, plot=False)

data = [ryt_avg, ryt_naive, ryt_drift, ryt_ses]
title = ["Average Method", "Naive Method", "Drift Method", "SES Method"]
plt_subplot(data, "ACF plot for 4 methods", title, 2, 2, "Magnitude", "Lag", acf=True, lag=lag)
