import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import chi2
import matplotlib.pyplot as plt
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()
np.random.seed(6313)
colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
    "#aec7e8",  # pale blue
    "#ffbb78",  # peach
    "#98df8a",  # yellowish green
    "#ff9896",  # coral
    "#c5b0d5",  # lavender
    "#c49c94",  # tan
    "#f7b6d2",  # bubblegum pink
    "#c7c7c7",  # light gray
    "#dbdb8d",  # greenish yellow
    "#9edae5"   # robin's egg blue
]


def cal_rolling_mean_var(y):
    n = len(y)
    rolling_mean = []
    rolling_var = []
    for i in range(n):
        mean = y[0:i].mean()
        var = np.var(y[0:i])  # .var()
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


def non_seasonal_differencing(y, order):
    diff = []
    for i in range(len(y)):
        if i < order:
            diff.append(0)
        else:
            # diff.append(df[f"{column}"][i] - df[f"{column}"][i - 1])
            diff.append(y[i] - y[i - 1])

    return diff


def seasonal_differencing(y, seasonal_period):
    n = len(y)
    y_seasonal_diff = np.zeros_like(y)  # create an array of zeros with the same shape as y
    for i in range(seasonal_period, n):
        y_seasonal_diff[i] = y[i] - y[i - seasonal_period]
    return y_seasonal_diff


def auto_corr(yt, lag, title=None, plot=True, marker_thickness=2, line_width=2):
    y_bar = np.mean(yt)
    ryt = []
    arr2 = []
    for k in range(len(yt)):
        y1 = yt[k] - y_bar
        arr2.append(y1 * y1)
    norm_var = np.sum(arr2)

    for i in range(lag + 1):
        arr1 = []
        for j in range(i, len(yt)):
            y1 = yt[j] - y_bar
            y2 = yt[j - i] - y_bar
            arr1.append(y1 * y2)

        ryt.append(round(np.sum(arr1) / norm_var, 4))
        del arr1

    if title is None:
        title = f'y = {yt}'

    if plot:
        ryt_plot = ryt[::-1] + ryt[1:]
        lags = [-x for x in range(lag, 0, -1)] + [x for x in range(lag + 1)]
        markerline, stemlines, baseline = plt.stem(lags, ryt_plot, markerfmt='red', basefmt='gray', linefmt='blue')
        plt.axhspan(-1.96 / np.sqrt(len(yt)), 1.96 / np.sqrt(len(yt)), color="lavender")
        plt.setp(stemlines, 'linewidth', line_width)
        plt.setp(markerline, 'markersize', marker_thickness)
        plt.xlabel("Lags")
        plt.ylabel("Magnitude")
        plt.title(f'{title}')
        plt.show()
    return ryt


def plt_subplot(data, plot_title, title, row, col, ylab, xlab, acf=False, lag=None, marker_thickness=2, line_width=2):
    # colors = ['chartreuse', 'olive', 'salmon', 'teal', 'plum', 'lavender', 'navy']
    color = random.choice(colors)
    fig, axes = plt.subplots(nrows=row, ncols=col)
    for i, ax in enumerate(axes.flat):
        if acf:
            ryt = data[i]
            ryt_plot = ryt[::-1] + ryt[1:]
            ryt_plot = [x for x in ryt_plot if x is not None]
            ryt_len = [x for x in ryt if x is not None]
            lags = [-x for x in range(lag, 0, -1)] + [x for x in range(lag + 1)]
            markerline, stemlines, baseline = ax.stem(lags, ryt_plot, markerfmt='red', basefmt='gray', linefmt='blue')
            plt.setp(stemlines, 'linewidth', line_width)
            plt.setp(markerline, 'markersize', marker_thickness)
            ax.axhspan(-1.96 / np.sqrt(len(ryt_len)), 1.96 / np.sqrt(len(ryt_len)), color="lavender")
            ax.set_title(f"{title[i]}")
            ax.set_ylabel(ylab)
            ax.set_xlabel(xlab)
        else:
            ax.plot(data[i], color=color)
            ax.set_title(f"{title[i]}")
            ax.tick_params(axis='x', labelsize=8)
            ax.grid()
            ax.set_ylabel(ylab)
            ax.set_xlabel(xlab)

    plt.tight_layout(h_pad=2, w_pad=2)
    plt.subplots_adjust(bottom=0.1, left=0.11, top=0.85)
    # fig.supxlabel(xlab)
    # fig.supylabel(ylab)
    fig.suptitle(plot_title)
    plt.show()


def box_pierce_test_q_value(y_hat, lag, t):
    """
    :param y_hat:
    :param lag:
    :param t: num of observations
    :return:
    """
    rk = auto_corr(y_hat, lag, plot=False)
    rk2 = [x ** 2 for x in rk[1:]]
    # rk[1:] because not including acf=1 as it will make Q-value big
    q = t * sum(rk2)
    return round(q, 3)

def average_method(yt, i, n_train):
    if i < n_train:
        y_hat = sum(yt[:i]) / (len(yt[:i]))
    else:
        i = n_train
        y_hat = sum(yt[:i]) / (len(yt[:i]))
    return y_hat


def naive_method(yt, i, n_train):
    if i < n_train:
        y_hat = yt[i - 1]
    else:
        y_hat = yt[n_train - 1]
    return y_hat


def drift_method(yt, i, n_train):
    if i <= 1:
        y_hat = np.nan
    elif i < n_train:
        y_hat = yt[i-1] + 1 * ((yt[i-1] - yt[0]) / (i-1))
    else:
        y_hat = yt[n_train-1] + (i-n_train+1) * ((yt[n_train-1] - yt[0]) / (n_train - 1))
    return y_hat


def simple_expo_smoothing_method(yt, yt_hat, i, n_train, alpha=0.5):
    if i == 0:
        y_hat = yt[i]
    elif i < n_train:
        y_hat = alpha * yt[i-1] + (1 - alpha) * yt_hat[i-1]
    else:
        y_hat = alpha * yt[n_train-1] + (1 - alpha) * yt_hat[n_train-1]  # yt_hat[n_train - 1]
    return y_hat


def backward_regression(x_train_s, x_train, Y):
    # scaler = StandardScaler()
    cols = x_train.columns
    X = sm.add_constant(x_train_s)
    model_orig = sm.OLS(Y, X).fit()
    aic_orig = model_orig.aic
    bic_orig = model_orig.bic
    adj_rsquare = model_orig.rsquared_adj
    criterions = {"aic": [], "bic": [], "adj_rsq": []}

    selected_feats = [[cols]]
    while True:
        for i, x in enumerate(cols):
            new_train = x_train
            new_X_df = new_train.drop(cols[i], axis=1)
            # new_X = scaler.fit_transform(new_X_df)
            X = sm.add_constant(new_X_df)
            new_model = sm.OLS(Y, X).fit()

            if aic_orig > new_model.aic or bic_orig > new_model.bic and adj_rsquare < new_model.rsquared_adj:
                selected_feats.append([new_X_df.columns])
                criterions['aic'].append(new_model.aic)
                criterions['bic'].append(new_model.bic)
                criterions['adj_rsq'].append(new_model.rsquared_adj)
                aic_orig, bic_orig, adj_rsquare = new_model.aic, new_model.bic, model_orig.rsquared_adj

            del new_train
            for j, y in enumerate(cols):
                if 0 < j < 13:
                    new_train = x_train
                    new_train = new_train.drop(cols[i], axis=1)
                    new_X_df = new_train.drop(new_train.columns[:j], axis=1)
                    # new_X = scaler.fit_transform(new_X_df)
                    X = sm.add_constant(new_X_df)
                    new_model = sm.OLS(Y, X).fit()

                    if aic_orig > new_model.aic or bic_orig > new_model.bic and \
                            adj_rsquare < new_model.rsquared_adj:
                        # print(i, [x for x in range(j)])
                        criterions['aic'].append(new_model.aic)
                        criterions['bic'].append(new_model.bic)
                        criterions['adj_rsq'].append(new_model.rsquared_adj)
                        selected_feats.append(new_X_df.columns)
                        aic_orig, bic_orig, adj_rsquare = new_model.aic, new_model.bic, model_orig.rsquared_adj

                    del new_train
        break

    index_value = criterions['adj_rsq'].index(max(criterions['adj_rsq']))
    return selected_feats[index_value]


def moving_average(y):

    m = int(input("Enter order of moving average (m): "))
    # assert m > 2, "m=1,2 will not be accepted"
    if m <= 2:
        print("m=1,2 will not be accepted")
        return 0
    if m % 2 == 0:
        mf = int(input("Enter folding order (second m): "))
        if mf % 2 != 0:
            print("Second m should be even")
            return 0
    k = (m - 1) / 2
    if m % 2 == 0:
        k = int(k + 0.5)
    t_hat_t = []
    for i, x in enumerate(y):
        if i < k or i > len(y)-k-1:
            t_hat_t.append(np.nan)
        else:
            if m % 2 == 0:
                # print(y[i-k:i+k])
                y_t_j = np.mean(y[i-k:i+k])
            else:
                k = int(k)
                # print(y[i - k:i + k + 1])
                y_t_j = np.mean(y[i - k:i + k + 1])
            t_hat_t.append(round(y_t_j, 2))

    t_hat_t_final = []
    if m % 2 == 0:
        # print(t_hat_t)
        kf = float((mf - 1) / 2)
        kf = int(kf + 0.5)
        for i, x in enumerate(t_hat_t):
            y_t_j = np.mean(t_hat_t[i - kf:i + kf])
            t_hat_t_final.append(round(y_t_j, 2))
    else:
        t_hat_t_final = t_hat_t

    detrend = []
    for a, b in zip(y, t_hat_t_final):
        resid = a - b
        detrend.append(round(resid, 2))
    return t_hat_t_final, detrend


def subplotting(xdata, ydata1, detrended, ydata2, plot_title, title, row, col, ylab, xlab, acf=False, lag=None, marker_thickness=2, line_width=2):
    color1, color2, color3 = random.choice(colors), random.choice(colors), random.choice(colors)
    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(14,8))
    # plt.figure()
    for i, ax in enumerate(axes.flat):
        ax.plot(xdata, ydata1[i], color=color1, label=f"{title[i]}")
        ax.plot(xdata, detrended[i], color=color2, label="Detrended")
        ax.plot(xdata, ydata2[i], color=color3, label="Original")
        ax.plot()
        ax.set_title(f"{title[i]}")
        ax.tick_params(axis='x', labelsize=8)
        ax.grid()
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        ax.legend()
        plt.tight_layout()
    # plt.subplots_adjust(bottom=0.1, left=0.11, top=0.85)
    fig.suptitle(plot_title)
    plt.show()


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


def forecast_plot_test(yt, t, e, e2, cap_yt, method, n_train, results=True, alpha=None, lag=5):
    # plt.figure(figsize=(10, 8))
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
    q_value = box_pierce_test_q_value(e[2:n_train], lag, n_train - 2)
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


def ar_ma_order_2(e, a1, a2, process):

    # e = np.random.normal(mean, std, N)
    y = np.zeros(len(e))

    k = y if process == "ar" else e
    for i in range(len(e)):
        if i == 0:
            y[0] = e[0]
        elif i == 1:
            y[i] = a1 * k[i - 1] + e[i]
        else:
            y[i] = a1 * k[i - 1] + a2 * k[i - 2] + e[i]

    return y


def ar_ma_dlsim(e, num, den, process):

    # e = np.random.normal(mean, std, N)
    system = (num, den, 1)# if process == "ar" else (den, num, 1)
    t, y_dlsim = signal.dlsim(system, e)

    return y_dlsim


def ar_process(mean, std):
    np.random.seed(6313)
    N = int(input("Enter number of samples (N): "))
    order = int(input("Enter order of AR process: "))
    e = np.random.normal(mean, std, N)

    num, den = [], []
    for i in range(order+1):
        n = 1 if i == 0 else 0
        d = 1 if i == 0 else float(input(f"Enter denominator (coeff) {i}: "))
        num.append(n), den.append(d)

    # system = (num, den, 1)
    # t, y_dlsim = signal.dlsim(system, e)
    y_dlsim = ar_ma_dlsim(e, num, den, "ar")

    y, na = y_dlsim, order
    T = N - na - 1
    X = np.zeros((T, na))
    for i in range(na, T + na):
        for j in range(na):
            X[i - na, j] = y[i - j - 1]
    Y = y[na:na + T]
    a_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    a_hat_orig = [-i for i in a_hat.flatten()]
    a_hat_round = [round(-i, 2) for i in a_hat.flatten()]

    return a_hat_orig, den[1:], na, N, a_hat_round


def sm_arma_process(lag):
    N = int(input("Enter number of samples (N): "))
    mean = float(input("Enter the mean of white noise: "))
    variance = float(input("Enter the variance of white noise: "))
    na = int(input("Enter AR order: "))
    nb = int(input("Enter MA order: "))
    # e = np.random.normal(mean, std, N)

    ar_coeffs, ma_coeffs = [], []
    print("Enter the coefficients of AR (separated by space):")
    ar_coeffs = [float(x) for x in input().split()]

    print("Enter the coefficients of MA (separated by space):")
    ma_coeffs = [float(x) for x in input().split()]

    ar_params = np.array(ar_coeffs)
    ma_params = np.array(ma_coeffs)
    arma_process = sm.tsa.ArmaProcess(ar_params, ma_params)
    # white_noise = np.sqrt(variance) * np.random.randn(N) + mean
    mean_y = mean * ((1 + np.sum(ma_params)) / (1 + np.sum(ar_params)))
    arma_data = arma_process.generate_sample(nsample=N, scale=np.sqrt(variance), distrvs=None, burnin=0,
                                             axis=0) + mean_y  # + white_noise
    ryt = arma_process.acf(lags=lag)
    # ry = ryt[::-1] + ryt[1:]
    return arma_data, ryt


def phi_j_kk(ryt, j, k):
    num = np.zeros((k, k))
    den = np.zeros((k, k))

    for col in range(k):    # k+1
        for row in range(k):    # row -> k
            if col == k - 1:
                num[row, col] = ryt[j + row + 1]
                den[row, col] = ryt[j - k + 1 + row]
            else:
                num[row, col] = ryt[j + row - col]  # row -> k-1
                den[row, col] = num[row, col]

    if np.linalg.det(den) != 0:
        phi = np.linalg.det(num) / np.linalg.det(den)
    else:
        phi = np.inf

    return round(phi, 3)

def gpac(ryt, na, nb):

    gpac_arr = np.zeros((nb, na-1))
    for k in range(1, na):
        for j in range(nb):
            gpac_arr[j, k-1] = phi_j_kk(ryt, j, k)

    return gpac_arr


def acf_pacf_plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=2)
    plt.show()


def error_analysis(yt, pred, n, s_o):
    error = []
    e_squared = []
    for i in range(0, len(yt)):
        if pred[i] == np.nan:
            error.append(np.nan)
            e_squared.append(np.nan)
        else:
            error.append(yt[i] - pred[i])
            e_squared.append(error[i] ** 2)

    mse_tr = np.nanmean(e_squared[s_o:n])
    mse_ts = np.nanmean(e_squared[n:])
    res_mean = np.nanmean(error[s_o:n])
    var_pred = np.nanvar(error[s_o:n])
    var_fcst = np.nanvar(error[n:])

    print(f'Mean of residual error is {np.round(res_mean, 2)}')
    print(f'MSE of residual error for  is {np.round(mse_tr, 2)}')
    print(f'Variance of residual error   is {np.round(var_pred, 2)}')
    print(f'Variance of forecast error  is {np.round(var_fcst, 2)}')
    print(f'Ratio of variance of residual errors versus variance of forecast errors : {np.round(var_pred / var_fcst, 2)}')

    return error, e_squared, mse_tr, mse_ts, var_pred, var_fcst, res_mean


def model_coefficients_and_intervals(model, na, nb):
    for i in range(na):
        print(f'The AR coefficient a{i} is: {-1 * model.params[i]}')
    for i in range(nb):
        print(f'The MA coefficient a{i} is: {model.params[i + na]}')

    for i in range(0, na):
        print(f"The confidence interval for a{i} is: {-model.conf_int()[0][i]} and {-model.conf_int()[1][i]}")
        zero_in_interval = ((-model.conf_int()[0][i] < 0) & (-model.conf_int()[1][i] > 0)).any()
        print("Zero in confidence interval: ", zero_in_interval)
        if zero_in_interval:
            print("Model is biased because of zero in confidence interval.")
        else:
            print("Model is unbiased.")
    for i in range(0, nb):
        print(f"The confidence interval for b{i} is: {model.conf_int()[0][i + na]} and {model.conf_int()[1][i + na]}")
        zero_in_interval = ((model.conf_int()[0][i + na] < 0) & (model.conf_int()[1][i + na] > 0)).any()
        print("Zero in confidence interval: ", zero_in_interval)
        if zero_in_interval:
            print("Model is biased because of zero in confidence interval.")
        else:
            print("Model is unbiased.")


def arima_model(na, nb, lags, y_train, y_test):
    model = sm.tsa.SARIMAX(y_train, order=(na, 0, 0), seasonal_order=(0, 0, nb, 24)).fit()
    print(model.summary())
    model_hat = model.predict()
    test_forecast = model.forecast(len(y_test))
    error_analysis(pd.concat([y_train, y_test]), pd.concat([model_hat, test_forecast]), len(y_train), 0)
    acf_pacf_plot(model.resid, lags)
    Q = sm.stats.acorr_ljungbox(model.resid, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for ARIMA residuals: ", Q)
    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)
    if Q < chi_critical:
        print("As Q-value is less than chi-2 critical, Residual is white")
    else:
        print("As Q-value is greater than chi-2 critical, Residual is NOT white")
    model_coefficients_and_intervals(model, na, nb)
    result = sm.stats.acorr_ljungbox(model.resid, model_df=1, boxpierce=True, lags=[20])
    print(result)
    plt.plot(list(y_train.index.values + 1), y_train, label='Training dataset')
    plt.plot(list(y_test.index.values + 1), y_test, label='Testing dataset', color='orange')
    plt.plot(list(y_test.index.values + 1), test_forecast, label='Forecast',  color='green')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('SARIMA')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('sarima.png')
    plt.show()


def compute_lm_step1(y, na, nb, delta, theta):
    n1 = na + nb
    e = calculate_error(y, na, theta)
    sse_old = np.dot(e.T, e)[0][0]
    X = np.empty((len(y), n1))
    for i in range(n1):
        theta[i] += delta
        e_i = calculate_error(y, na, theta)
        x_i = (e - e_i) / delta
        X[:, i] = x_i[:, 0]
        theta[i] -= delta
    A = np.dot(X.T, X)
    g = np.dot(X.T, e)
    return A, g, X, sse_old


def compute_lm_step2(y, na, A, theta, mu, g):
    delta_theta = np.linalg.solve(A + mu * np.eye(A.shape[0]), g)
    theta_new = theta + delta_theta
    e_new = calculate_error(y, na, theta_new)
    sse_new = np.dot(e_new.T, e_new)[0][0]
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    return sse_new, delta_theta, theta_new


def compute_lm_step3(y, na, nb):
    N = len(y)
    n = na + nb
    mu = 0.01
    mu_max = 10 ** 20
    max_iterations = 100
    delta = 10 ** -6
    var_e = 0
    covariance_theta_hat = 0
    sse_list = []
    theta = np.zeros((n, 1))

    for iterations in range(max_iterations):
        A, g, X, sse_old = compute_lm_step1(y, na, nb, delta, theta)
        sse_new, delta_theta, theta_new = compute_lm_step2(y, na, A, theta, mu, g)
        sse_list.append(sse_old)
        if iterations < max_iterations:
            if sse_new < sse_old:
                if np.linalg.norm(delta_theta, 2) < 10 ** -3:
                    theta_hat = theta_new
                    var_e = sse_new / (N - n)
                    covariance_theta_hat = var_e * np.linalg.inv(A)
                    print("Convergence Occured in ", iterations)
                    break
                else:
                    theta = theta_new
                    mu /= 10
            while sse_new >= sse_old:
                mu = mu * 10
                if mu > mu_max:
                    print('No Convergence')
                    break
                sse_new, delta_theta, theta_new = compute_lm_step2(y, na, A, theta, mu, g)
        else:
            print('Max Iterations Reached')
            break
        theta = theta_new
    return theta_new, sse_new, var_e, covariance_theta_hat, sse_list


def lm_confidence_interval(theta, cov, na, nb):
    print("Confidence Interval for the Estimated Parameters")
    lower_bound = theta - 2 * np.sqrt(np.diag(cov))
    upper_bound = theta + 2 * np.sqrt(np.diag(cov))
    round_off = 3
    lower_bound = np.round(lower_bound, decimals=round_off)
    upper_bound = np.round(upper_bound, decimals=round_off)
    for i in range(na + nb):
        if i < na:
            print(f"AR Coefficient {i+1}: ({lower_bound[i][0]}, {upper_bound[i][0]})")
        else:
            print(f"MA Coefficient {i + 1 - na}: ({lower_bound[i][0]}, {upper_bound[i][0]})")


def calculate_error(y, na, theta):
    np.random.seed(6313)
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):
        for x in range(len(den) - len(num)):
            num = np.append(num, 0)
    elif len(num) > len(den):
        for x in range(len(num) - len(den)):
            den = np.append(den, 0)
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    return e


def find_roots(theta, na):
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):
        num = np.pad(num, (0, len(den) - len(num) - 1), mode='constant')
    elif len(num) > len(den):
        den = np.pad(den, (0, len(num) - len(den) - 1), mode='constant')
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    roots_num = np.round(np.roots(num), decimals=3)
    roots_den = np.round(np.roots(den), decimals=3)
    print("Poles :", roots_num)
    print("Zeros:", roots_den)


def plot_sse(sse_list):
    plt.plot(sse_list)
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.title('SSE over the iterations')
    plt.savefig("sse.png")
    plt.show()


def lm(y, na, nb):
    theta, sse, var_error, cov_theta_hat, sse_list = compute_lm_step3(y, na, nb)
    theta2 = np.array(theta).reshape(-1)
    for i in range(na + nb):
        if i < na:
            ar_coef = "{:.3f}".format(theta2[i])
            print(f"The AR coefficient {i + 1} is: {ar_coef}")
        else:
            ma_coef = "{:.3f}".format(theta2[i])
            print(f"The MA coefficient {i - na + 1} is: {ma_coef}")

    lm_confidence_interval(theta, cov_theta_hat, na, nb)
    cov_theta_hat_rounded = np.round(cov_theta_hat, decimals=3)
    print("Estimated Covariance Matrix of estimated parameters:")
    print(cov_theta_hat_rounded)
    var_error_rounded = round(var_error, 3)
    print("Estimated variance of error:", var_error_rounded)
    find_roots(theta, na)
    plot_sse(sse_list)




