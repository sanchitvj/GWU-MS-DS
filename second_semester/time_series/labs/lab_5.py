import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from toolbox import sm_arma_process, gpac, ACF_PACF_Plot

np.random.seed(6313)

# y, ryt = sm_arma_process(lag=15)
# print([round(i, 2) for i in y[:10]])
#
# gpac_arr = gpac(ryt, 7, 7)
#
# sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[1, 2, 3, 4, 5, 6])
# plt.title("GPAC Table")
# plt.show()
#
# ACF_PACF_Plot(y, 20)

#########################################
# ARMA (0,1) :   y(t) =   e(t) + 0.5e(t1)
#########################################

y, ryt = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

gpac_arr = gpac(ryt, 10, 7)

sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[1, 2, 3, 4, 5, 6])
plt.title("GPAC Table")
plt.show()

ACF_PACF_Plot(y, 50)

# ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)

# ARMA (2,0): y(t) + 0 .5y(t-1) + 0.2y(t-2) = e(t)

# ARMA (2,1):  y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)

# ARMA (1,2):  y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)

# ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)

# ARMA (2,2): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) + 0.5e(t-1) - 0.4e(t-2)

# ARMA (13,0): yt - 0.5y(t-1) -0.2 y(t-12) + 0.1y(t-13) = e(t)
