import random
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas_datareader import data
import yfinance as yf

yf.pdr_override()


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
    colors = ['chartreuse', 'olive', 'salmon', 'teal', 'plum', 'lavender', 'navy']
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
    # n = len(yt)
    # slope = (yt[n - 1] - yt[0]) / (n - 1)
    # intercept = yt[0] - (slope * 1)
    # y_hat = slope * (i + 1) + intercept  # for t in range(1, n + 1)
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
    # print(cols[1:8])
    X = sm.add_constant(x_train_s)
    model_orig = sm.OLS(Y, X).fit()
    aic_orig = model_orig.aic
    bic_orig = model_orig.bic
    adj_rsquare = model_orig.rsquared_adj
    # criterions = {"aic": [aic_orig], "bic": [bic_orig], "adj_rsq": [adj_rsquare]}
    criterions = {"aic": [], "bic": [], "adj_rsq": []}

    selected_feats = [[cols]]
    while True:
        for i, x in enumerate(cols):
            # if i == 0:
            new_train = x_train
            new_X_df = new_train.drop(cols[i], axis=1)
            new_X = scaler.fit_transform(new_X_df)
            X = sm.add_constant(new_X)
            new_model = sm.OLS(Y, X).fit()

            if aic_orig > new_model.aic or bic_orig > new_model.bic and adj_rsquare < new_model.rsquared_adj:
                # criterions['aic'][0] = new_model.aic
                # criterions['bic'][0] = new_model.bic
                # criterions['adj_rsq'][0] = new_model.rsquared_adj
                selected_feats.append([new_X_df.columns])
                criterions['aic'].append(new_model.aic)
                criterions['bic'].append(new_model.bic)
                criterions['adj_rsq'].append(new_model.rsquared_adj)
                aic_orig, bic_orig, adj_rsquare = new_model.aic, new_model.bic, model_orig.rsquared_adj
            # else:
            #     break
            # print(i)
            # else:
            del new_train
            for j, y in enumerate(cols):
                # if j == 0:
                #     continue
                if 0 < j < 13:
                    new_train = x_train
                    # for k, z in enumerate(cols):
                    # print(cols[1:j])
                    new_train = new_train.drop(cols[i], axis=1)
                    new_X_df = new_train.drop(new_train.columns[:j], axis=1)
                    new_X = scaler.fit_transform(new_X_df)
                    X = sm.add_constant(new_X)
                    new_model = sm.OLS(Y, X).fit()

                    # v = new_model.rsquared_adj
                    # print(j)
                    if aic_orig > new_model.aic or bic_orig > new_model.bic and \
                            adj_rsquare < new_model.rsquared_adj:
                        # criterions['aic'][0] = new_model.aic
                        # criterions['bic'][0] = new_model.bic
                        # criterions['adj_rsq'][0] = new_model.rsquared_adj
                        print(i, [x for x in range(j)])
                        criterions['aic'].append(new_model.aic)
                        criterions['bic'].append(new_model.bic)
                        criterions['adj_rsq'].append(new_model.rsquared_adj)
                        selected_feats.append(new_X_df.columns)
                        aic_orig, bic_orig, adj_rsquare = new_model.aic, new_model.bic, model_orig.rsquared_adj

                    # else:
                    #     # print("breaking")
                    #     break
                    del new_train
        break
    index_value = criterions['adj_rsq'].index(max(criterions['adj_rsq']))
    # print(criterions['adj_rsq'][index_value], index_value)
    return selected_feats[index_value]



