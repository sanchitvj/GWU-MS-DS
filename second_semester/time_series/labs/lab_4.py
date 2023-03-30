import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from toolbox import auto_corr, ar_ma_order_2, ar_ma_dlsim,\
    ar_process, adf_test, kpss_test
import warnings
warnings.filterwarnings('ignore')
np.random.seed(6313)

mean = 0
std = 1
N = 1000
e = np.random.normal(mean, std, N)

y_ar = ar_ma_order_2(e, 0.5, 0.2, "ar")

plt.plot(y_ar)
plt.title("AR process: y(t)−0.5y(t−1)−0.2y(t−2)=e(t)")
plt.xlabel("Samples")
plt.ylabel("y")
plt.show()

lags = 20
ryt = auto_corr(y_ar, lag=lags, title="ACF of y AR process")

print("ADF and KPSS test for AR process")
adf_test(y_ar)
kpss_test(y_ar)

y_ar = [round(i, 2) for i in y_ar]
print(f'For loop Method AR process: {y_ar[:5]}')

num = [1, 0, 0]
den = [1, -0.5, -0.2]
y_dlsim = ar_ma_dlsim(e, num, den, "ar")
# system = (num, den, 1)
# e = np.random.normal(0, 1, 1000)
# t, y_dlsim = signal.dlsim(system, e)
y_dlsim = [round(i, 2) for i in y_dlsim.flatten()]
print(f'y(dlsim) for AR: {y_dlsim[:5]}')

a_hat_1k, coeff, order, samples, a_hat = ar_process(0, 1)
print(f"Coefficient and a_hat values for order {order} and number of samples {samples} are:")
print("Original coefficients: ", coeff)
print("Estimated coefficients (a_hat): ", a_hat)

a_hat_5k, coeff, order, samples, a_hat = ar_process(0, 1)
print(f"Coefficient and a_hat values for order {order} and number of samples {samples} are:")
print("Original coefficients: ", coeff)
print("Estimated coefficients (a_hat): ", a_hat)

a_hat_10k, coeff, order, samples, a_hat = ar_process(0, 1)
print(f"Coefficient and a_hat values for order {order} and number of samples {samples} are:")
print("Original coefficients: ", coeff)
print("Estimated coefficients (a_hat): ", a_hat)

# for x, y, z in zip(a_hat_1k, a_hat_5k, a_hat_10k):
mse_1k = ((coeff[0] - a_hat_1k[0])**2 + (coeff[1] - a_hat_1k[1])**2) / 2
mse_5k = ((coeff[0] - a_hat_5k[0])**2 + (coeff[1] - a_hat_5k[1])**2) / 2
mse_10k = ((coeff[0] - a_hat_10k[0])**2 + (coeff[1] - a_hat_10k[1])**2) / 2

print("Root Mean squared error for 1000 samples: ", np.sqrt(mse_1k))
print("Root Mean squared error for 5000 samples: ", np.sqrt(mse_5k))
print("Root Mean squared error for 10000 samples: ", np.sqrt(mse_10k))

a_hat_1k, coeff, order, samples, a_hat = ar_process(0, 1)
print(f"Coefficient and a_hat values for order {order} and number of samples {samples} are:")
print("Original coefficients: ", coeff)
print("Estimated coefficients (a_hat): ", a_hat)

a_hat_10k, coeff, order, samples, a_hat = ar_process(0, 1)
print(f"Coefficient and a_hat values for order {order} and number of samples {samples} are:")
print("Original coefficients: ", coeff)
print("Estimated coefficients (a_hat): ", a_hat)

a_hat_100k, coeff, order, samples, a_hat = ar_process(0, 1)
print(f"Coefficient and a_hat values for order {order} and number of samples {samples} are:")
print("Original coefficients: ", coeff)
print("Estimated coefficients (a_hat): ", a_hat)

# def mse_coeff_a_hat(order, coeff, a_hat)
mse_1k, mse_10k, mse_100k = [], [], []
for i in range(order):
    mse_1k.append((coeff[i] - a_hat_1k[i])**2)
    mse_10k.append((coeff[i] - a_hat_10k[i])**2)
    mse_100k.append((coeff[i] - a_hat_100k[i])**2)

print("Root Mean squared error for 1000 samples: ", np.sqrt(np.mean(mse_1k)))
print("Root Mean squared error for 5000 samples: ", np.sqrt(np.mean(mse_10k)))
print("Root Mean squared error for 10000 samples: ", np.sqrt(np.mean(mse_100k)))


y_ma = ar_ma_order_2(e, 0.5, 0.2, "ma")

plt.plot(y_ma)
plt.title("MA process: y(t)=e(t)+0.5e(t−1)+0.2e(t−2)")
plt.xlabel("Samples")
plt.ylabel("y")
plt.show()

ryt = auto_corr(y_ma, lags, "ACF of y MA process")

e_10k = np.random.normal(mean, std, 10000)
y_ma_10k = ar_ma_order_2(e_10k, 0.5, 0.2, "ma")
ryt = auto_corr(y_ma_10k, lags, "ACF of y MA process 10000 samples")
e_100k = np.random.normal(mean, std, 100000)
y_ma_100k = ar_ma_order_2(e_100k, 0.5, 0.2, "ma")
ryt = auto_corr(y_ma_100k, lags, "ACF of y MA process 100000 samples")

print("ADF and KPSS test for MA process")
adf_test(y_ma)
kpss_test(y_ma)

y_ma = [round(i, 2) for i in y_ma]
print(f'For loop Method MA process: {y_ma[:5]}')

# e = np.random.normal(mean, std, 1000)
y_dlsim = ar_ma_dlsim(e, [1, 0, 0], [1, 0.5, 0.2], "ma")
y_dlsim = [round(i, 2) for i in y_dlsim.flatten()]
print(f'y(dlsim) for MA: {y_dlsim[:5]}')
