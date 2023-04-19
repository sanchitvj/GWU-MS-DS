import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pprint import pprint
from toolbox import sm_arma_process, gpac, acf_pacf_plot

np.random.seed(6313)

#####################################
# 𝑦(𝑡)−0.5𝑦(𝑡−1)=𝑒(𝑡)
#####################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#########################################
# ARMA (0,1) :   y(t) = e(t) + 0.5e(t1)
#########################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#####################################
# ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)
#####################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#####################################
# ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
#####################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#####################################
# ARMA (2,1):  y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
#####################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#####################################
# ARMA (1,2):  y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)
#####################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#####################################
# ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)
#####################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#####################################
# ARMA (2,2): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) + 0.5e(t-1) - 0.4e(t-2)
#####################################
y, ryt, na, nb = sm_arma_process(lag=15)
print([round(i, 2) for i in y[:10]])

j, k = 7, 7
gpac_arr = gpac(ryt, j, k)

heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
plt.title("GPAC Table")
plt.show()

acf_pacf_plot(y, 20)

#####################################
# ARMA (13,0): yt - 0.5y(t-1) -0.2 y(t-12) + 0.1y(t-13) = e(t)
#####################################
# y, ryt, na, nb = sm_arma_process(lag=15)
# print([round(i, 2) for i in y[:10]])
#
# j, k = 7, 7
# gpac_arr = gpac(ryt, j, k)
#
# heatmap = sns.heatmap(gpac_arr, annot=True, linewidths=1, xticklabels=[i for i in range(1, 7)])
# heatmap.add_patch(Rectangle((na-1, nb), 1, j-nb, fill=False, edgecolor='green', lw=4))  # j line vertical
# heatmap.add_patch(Rectangle((na, nb+1), 1, k-1-na, fill=False, angle=270, edgecolor='yellow', lw=4))  # k line horizontal
# plt.title("GPAC Table")
# plt.show()
#
# acf_pacf_plot(y, 20)
