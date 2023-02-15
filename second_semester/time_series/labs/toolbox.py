import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

def cal_rolling_mean_var(df, column):
    n = len(df)
    rolling_mean = []
    rolling_var = []
    for i in range(n):
        mean = df[f"{column}"][0:i].mean()
        var = np.var(df[f"{column}"][0:i])  # .var()
        # use the np.var() function when you want to calculate the population
        # variance and use the .var() function when you want to calculate the sample variance.
        rolling_mean.append(mean)
        rolling_var.append(var)

    return rolling_mean, rolling_var


def adf_test(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        if int(key.strip('%')) <= 5:
            print('\t%s: %.3f' % (key, value))


def kpss_test(timeseries):
    # print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    # kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    print(f"KPSS Statistic: {kpsstest[0]}")
    print(f"p-value: {kpsstest[1]}")
    print('Critical Values:')
    for key, value in kpsstest[3].items():
        if float(key.strip('%')) <= 5:
            # kpss_output['Critical Value (%s)'%key] = value
            # print (kpss_output)
            print('\t%s: %.3f' % (key, value))


def non_seasonal_differencing(df, column, order):
    diff = []
    for i in range(len(df)):
        if i < order:
            diff.append(0)
        else:
            diff.append(df[f"{column}"][i] - df[f"{column}"][i - 1])

    return diff


def auto_corr(yt, lag, title=None):
    y_bar = np.mean(yt)
    ryt = []
    arr2 = []
    for k in range(len(yt)):
        y1 = yt[k] - y_bar
        arr2.append(y1 * y1)
    norm_var = np.sum(arr2)

    for i in range(lag+1):
        arr1 = []
        for j in range(i, len(yt)):
            y1 = yt[j] - y_bar
            y2 = yt[j - i] - y_bar
            arr1.append(y1 * y2)

        ryt.append(round(np.sum(arr1) / norm_var, 4))
        del arr1

    if title is None:
        title = f'y = {yt}'

    ryt_plot = ryt[::-1] + ryt[1:]
    lags = [-x for x in range(lag, 0, -1)] + [x for x in range(lag+1)]
    plt.stem(lags, ryt_plot, markerfmt='red', basefmt='gray', linefmt='blue')
    plt.axhspan(-1.96/np.sqrt(len(yt)), 1.96/np.sqrt(len(yt)), color="lavender")
    plt.xlabel("Lags")
    plt.ylabel("Magnitude")
    plt.title(f'Auto-correlation Function of {title}')
    plt.show()
    return ryt

