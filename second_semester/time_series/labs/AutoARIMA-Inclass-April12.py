from sktime.forecasting.arima import AutoARIMA
import numpy as np
import statsmodels.api as sm
from scipy import signal
import matplotlib.pyplot as plt

import sys
sys.path.append('C:\GW\Time series Analysis/toolbox')
from toolbox import check_num_den_size, correlation_coefficent_cal, LMA, Cal_autocorr, Cal_GPAC, difference, ADF_Cal, ACF_PACF_Plot, difference, Plot_Rolling_Mean_Var, check_num_den_size, ADF_Cal, kpss_test
from scipy.stats import chi2

# ARAM(2,2)
# y(t) - 0.5y(t-1) + 0.25y(t-2) = e(t) + 0.2e(t-1) + 0.5e(t-2)
bn = [1, 0.2, 0.5]
an = [1,-0.5, 0.25]

#== Able to find the order for the following example SARIMA(2,0,0)_3
# y(t) - 0.5y(t-3) + 0.25y(t-6) = e(t)
#========================================================
# bn = [1]
# an = [1,0,0,-0.5,0,0, 0.25]


# bn = [1]
# an = [1,0,0,-1.5,0,0, 0.5]
an, bn = check_num_den_size(an,bn)
system = (bn,an,1)
e = np.random.normal(0, 1, size=10000)
tout, y_new = signal.dlsim(system, e)
y = np.ndarray.flatten(y_new)
Plot_Rolling_Mean_Var(y, 'Raw Data')

y_train = y[:round(len(y)*.95)]
y_test = y[round(len(y)*.95):]
forecaster = AutoARIMA(start_p = 0,
                       max_p = 40,
                       start_q = 0,
                       max_q = 40,
                       start_P = 0,
                       max_P = 40,
                       start_Q = 0,
                       max_Q = 40,
                       seasonal = True,
                        max_d = 5,
                        max_D = 5,
                       # supress_warnings =True,
                       stationary = True,
                        n_fits = 20,
                       stepwise = False)

forecaster = forecaster.fit(y_train)
print(forecaster.summary())