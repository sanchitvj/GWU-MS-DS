import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from toolbox import box_pierce_test_q_value, average_method,\
    naive_method, drift_method, simple_expo_smoothing_method

yt_1 = [112, 118, 132, 129, 121, 135, 148, 136, 119, 104, 118, 115, 126, 141]
t = [x + 1 for x in range(14)]
n_train = 9
alpha = 0.25

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


def forecast_plot_test(yt, t, cap_yt, method, alpha=None):
    if method != "SES":
        plt.plot(t[:10], yt[:10], color='blue', label="training dataset")
        plt.plot(t[9:], yt[9:], color='orange', label="test dataset")
        plt.plot(t[9:], cap_yt[9:], color='green', label=f"{method} method h-step prediction")
        plt.title(f"{method} Method Forecast")
    else:
        plt.plot(t, yt, color='blue', label="original")
        plt.plot(t[:10], cap_yt[:10], color='orange', label=f"{method} 1-step prediction")
        plt.plot(t[9:], cap_yt[9:], color='green', label=f"{method} h-step prediction")
        plt.title(f"{method} Method Forecast with alpha = {alpha}")
    plt.ylabel("yt")
    plt.xlabel("time")
    plt.legend(loc='lower left')
    plt.show()

    print("\n")
    print(f"Results using {method} method forecast: ")
    print("y_hat_t: ", cap_yt)
    print("Error: ", e)
    print("Error-squared: ", e2)
    print("MSE training: ", round(np.nanmean(e2[:9]), 3))
    print("MSE testing: ", round(np.nanmean(e2[9:]), 3))
    print("Variance training: ", round(np.nanvar(e[:9]), 3))
    print("Variance testing: ", round(np.nanvar(e[9:]), 3))

    tn = 9  # training observations
    lag = 5
    box_pierce_test_q_value(cap_yt[1:9], lag, tn)


yt_hat, e, e2 = forecasting("average", yt_1, n_train)
forecast_plot_test(yt_1, t, yt_hat, "Average")

yt_hat, e, e2 = forecasting("naive", yt_1, n_train)
forecast_plot_test(yt_1, t, yt_hat, "Naive")

yt_hat, e, e2 = forecasting("drift", yt_1, n_train)
forecast_plot_test(yt_1, t, yt_hat, "Drift")

yt_hat, e, e2 = forecasting("ses", yt_1, n_train, alpha)
forecast_plot_test(yt_1, t, yt_hat, "SES", alpha)
