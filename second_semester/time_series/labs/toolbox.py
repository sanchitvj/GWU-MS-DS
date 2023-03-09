import random
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    scaler = StandardScaler()
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
            new_X = scaler.fit_transform(new_X_df)
            X = sm.add_constant(new_X)
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
                    new_X = scaler.fit_transform(new_X_df)
                    X = sm.add_constant(new_X)
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
            # if x is np.nan:
            #     # continue
            #     t_hat_t_final.append(x)
            # else:
            #     if i - k == np.nan or i + k == np.nan:  # i < k or i > len(y) - k - 1:
            #         t_hat_t.append(np.nan)
            #     else:
            # print(t_hat_t[i - kf:i + kf])
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
    # colors = ['chartreuse', 'olive', 'salmon', 'teal', 'plum', 'lavender', 'navy']
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

