import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import ar_ma_dlsim, auto_corr
np.random.seed(6313)

mean, std, N = 2, 1, 100
e_orig = np.random.normal(mean, std, N)
means, var, samples, th_mean, th_var = [], [], ["100", "1000", "10000"], ["6.66", "6.66", "6.66"], ["1.71", "1.71", "1.71"]
df = pd.DataFrame(index=samples, columns=["Mean", "Variance", "Theoretical Mean", "Theoretical Variance"])
df["Theoretical Mean"] = th_mean
df["Theoretical Variance"] = th_var

##################################
# 𝑦(𝑡)−0.5𝑦(𝑡−1)−0.2𝑦(𝑡−2)=𝑒(𝑡)
##################################
num = [1, 0, 0]
den = [1, -0.5, -0.2]
y_dlsim_100 = ar_ma_dlsim(e_orig, num, den, 'ar')
y_dlsim = [round(i, 4) for i in y_dlsim_100.flatten()]
print(f'y(dlsim) for AR, 100 samples: {y_dlsim[:5]}')
print("The experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) for 100 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) for 100 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))
print("\n")

e = np.random.normal(mean, std, 1000)
y_dlsim = ar_ma_dlsim(e, num, den, 'ar')
y_dlsim = [round(i, 4) for i in y_dlsim.flatten()]
print(f'y(dlsim) for AR, 1000 samples: {y_dlsim[:5]}')
print("The experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) for 1000 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) for 1000 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))
print("\n")

e = np.random.normal(mean, std, 10000)
y_dlsim = ar_ma_dlsim(e, num, den, 'ar')
y_dlsim = [round(i, 4) for i in y_dlsim.flatten()]
print(f'y(dlsim) for AR, 10000 samples: {y_dlsim[:5]}')
print("The experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) for 10000 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) for 10000 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))

df["Mean"], df["Variance"] = means, var
print(df.to_string())
print("\n")
del means, var, df, samples, th_mean, th_var
ryt = auto_corr(y_dlsim_100, 20, "y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t); 20 lags")
ryt = auto_corr(y_dlsim_100, 40, "y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t); 40 lags")
ryt = auto_corr(y_dlsim_100, 80, "y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t); 80 lags", line_width=1)


##################################
# 𝑦(𝑡)=𝑒(𝑡)+0.1𝑒(𝑡−1)+0.4𝑒(𝑡−2)
##################################
means, var, samples, th_mean, th_var = [], [], ["100", "1000", "10000"], ["3", "3", "3"], ["1.17", "1.17", "1.17"]
df = pd.DataFrame(index=samples, columns=["Mean", "Variance", "Theoretical Mean", "Theoretical Variance"])
df["Theoretical Mean"] = th_mean
df["Theoretical Variance"] = th_var

num = [1, 0.1, 0.4]
den = [1, 0, 0]
y_dlsim_100 = ar_ma_dlsim(e_orig, num, den, 'ma')
y_dlsim = [round(i, 4) for i in y_dlsim_100.flatten()]
print(f'y(dlsim) for MA, 100 samples: {y_dlsim[:5]}')
print("The experimental Mean of y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2) for 100 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2) for 100 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))
print("\n")

e = np.random.normal(mean, std, 1000)
y_dlsim = ar_ma_dlsim(e, num, den, 'ma')
y_dlsim = [round(i, 4) for i in y_dlsim.flatten()]
print(f'y(dlsim) for MA, 1000 samples: {y_dlsim[:5]}')
print("The experimental Mean of y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2) for 1000 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2) for 1000 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))
print("\n")

e = np.random.normal(mean, std, 10000)
y_dlsim = ar_ma_dlsim(e, num, den, 'ma')
y_dlsim = [round(i, 4) for i in y_dlsim.flatten()]
print(f'y(dlsim) for MA, 10000 samples: {y_dlsim[:5]}')
print("The experimental Mean of y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2) for 10000 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2) for 10000 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))

df["Mean"], df["Variance"] = means, var
print(df.to_string())
print("\n")
del means, var, df, samples, th_mean, th_var
ryt = auto_corr(y_dlsim_100, 20, "y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2); 20 lags")
ryt = auto_corr(y_dlsim_100, 40, "y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2); 40 lags")
ryt = auto_corr(y_dlsim_100, 80, "y(t)=e(t) + 0.1e(t−1) + 0.4e(t−2); 80 lags", line_width=1)


#################################################
# 𝑦(𝑡)−0.5𝑦(𝑡−1)−0.2𝑦(𝑡−2)=𝑒(𝑡)+0.1𝑒(𝑡−1)+0.4𝑒(𝑡−2)
#################################################
means, var, samples, th_mean, th_var = [], [], ["100", "1000", "10000"], ["10", "10", "10"], ["3", "3", "3"]
df = pd.DataFrame(index=samples, columns=["Mean", "Variance", "Theoretical Mean", "Theoretical Variance"])
df["Theoretical Mean"] = th_mean
df["Theoretical Variance"] = th_var

num = [1, 0.1, 0.4]
den = [1, -0.5, -0.2]
y_dlsim_100 = ar_ma_dlsim(e_orig, num, den, 'ar')
y_dlsim = [round(i, 4) for i in y_dlsim_100.flatten()]
print(f'y(dlsim) for ARMA(2,2), 100 samples: {y_dlsim[:5]}')
print("The experimental Mean of ARMA(2,2) for 100 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of ARMA(2,2) for 100 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))
print("\n")

e = np.random.normal(mean, std, 1000)
y_dlsim = ar_ma_dlsim(e, num, den, 'ar')
y_dlsim = [round(i, 4) for i in y_dlsim.flatten()]
print(f'y(dlsim) for ARMA(2,2), 1000 samples: {y_dlsim[:5]}')
print("The experimental Mean of ARMA(2,2) for 1000 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of ARMA(2,2) for 1000 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))
print("\n")

e = np.random.normal(mean, std, 10000)
y_dlsim = ar_ma_dlsim(e, num, den, 'ar')
y_dlsim = [round(i, 4) for i in y_dlsim.flatten()]
print(f'y(dlsim) for ARMA(2,2), 10000 samples: {y_dlsim[:5]}')
print("The experimental Mean of ARMA(2,2) for 10000 samples is: ", round(np.mean(y_dlsim), 4))
print("The experimental Variance of ARMA(2,2) for 10000 samples is: ", round(np.var(y_dlsim), 4))
means.append(round(np.mean(y_dlsim), 4)), var.append(round(np.var(y_dlsim), 4))

df["Mean"], df["Variance"] = means, var
print(df.to_string())
print("\n")
del means, var, df, samples, th_mean, th_var

ryt = auto_corr(y_dlsim_100, 20, "ARMA(2,2); 20 lags")
ryt = auto_corr(y_dlsim_100, 40, "ARMA(2,2); 40 lags")
ryt = auto_corr(y_dlsim_100, 80, "ARMA(2,2); 80 lags", line_width=1)
