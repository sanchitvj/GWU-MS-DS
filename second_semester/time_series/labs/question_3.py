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

df_orig = pd.read_csv("question3.csv")
print(df_orig.shape)

df = df_orig.copy()
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

res_kpss = kpss_test(df["y"])
res_adf = adf_test(df["y"])
plot_rolling_mean_var(df["y"])

print("Making dataset stationary 1st order seasonal differencing:")
y_diff_1 = non_seasonal_differencing(df.y, 1)
res_kpss = kpss_test(y_diff_1)
res_adf = adf_test(y_diff_1)
plot_rolling_mean_var(y_diff_1)
#
y = y_diff_1
#
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

model = sm.tsa.ARIMA(y, order=(2, 0, 1))  # ARIMA model with specified order
results = model.fit()  # method_kwargs={'warn_convergence': False})  # Fit the ARIMA model
print(results.summary())


df["y_diff"] = y
df["y_diff"].plot()
df["y"].plot()
plt.legend()
plt.title("Raw and differenced data")
plt.xlabel("time")
plt.ylabel("y values")
plt.show()
