
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from toolbox import auto_corr, cal_rolling_mean_var, non_seasonal_differencing
np.random.seed(6313)
#
# mean = 0
# std = 1
# N = 1000
# e = np.random.normal(mean, std, N)
# # method 1
# y = np.zeros(len(e))
# for i in range(len(e)):
#     if i == 0:
#         y[0] = e[0]
#     elif i == 1:
#         y[i] = -0.5 * y[i-1] + e[i]
#     else:
#         y[i] = -0.5 * y[i-1] + e[i] - 0.25 * y[i-2]
# print(y[:3])
# num = [1, 0, 0]
# den = [1, 0.5, 0.25]
# system = (num, den, 1)
# t, y_dlsim = signal.dlsim(system, e)
# print(y_dlsim[:3])

mean_e = 0
var_e = 1
N = 1000
e = np.random.normal(mean_e, var_e, N)
num = [1, 0]
den = [1, 0.5]
system = (num, den, 1)
_, y = signal.dlsim(system, e)
print(f"the experimental variance of y is {np.var(y):.4f}")

ryt = auto_corr(y, 20, "ACF")

num = [1, 0.25]
den = [1, 0]
system = (num, den, 1)
_, y = signal.dlsim(system, e)
print(f"the experimental variance of y is {np.var(y):.4f}")

# ryt = auto_corr(y, 20, "ACF")

num = [1, 0.25]
den = [1, 0.5]
system = (num, den, 1)
_, y = signal.dlsim(system, e)
print(f"the experimental variance of y is {np.var(y):.4f}")

# ryt = auto_corr(y, 20, "ACF")

num = [1, -0.25, 0]
den = [1, -1.5, 0.5]
system = (num, den, 1)
_, y = signal.dlsim(system, e)
print(f"the experimental variance of y is {np.var(y):.4f}")
plt.plot(y)
plt.show()
# ryt = auto_corr(y, 20, "ACF")

# cal_rolling_mean_var(y)


num = [1, -0.25, 0, 0]
den = [1, -2.5, 2, -0.5]
system = (num, den, 1)
_, y = signal.dlsim(system, e)
print(f"the experimental variance of y is {np.var(y):.4f}")
plt.plot(y)
plt.show()
ryt = auto_corr(y, 20, "ACF")

rol_mean, rol_var = cal_rolling_mean_var(y)
plt.plot(rol_mean)
plt.show()

df = pd.date_range(start='1/1/2018', freq=len())
