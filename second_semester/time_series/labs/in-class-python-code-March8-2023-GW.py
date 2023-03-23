import matplotlib.pyplot as plt
import numpy as np
np.random.seed(6313)
#======================================
# y(t) + 0.5y(y-1) + 0.25 y(t-2) = e(t)
#======================================
mean = 1
std = 1
N = 1000
e = np.random.normal(mean, std, N)
#==============================================
# Method 1 : Simulate AR process using for loop
#===============================================
y = np.zeros(len(e))
for i in range(len(e)):
    if i == 0:
       y[0] = e[0]

    elif i ==1:
        y[i] = -0.5*y[i-1] + e[i]

    else:
        y[i] = -0.5*y[i-1]  -0.25*y[i-2] + e[i]
print(f'For loop Method {y[:3]}')

#==============================================
# Method 2 : dlsim method
#===============================================
from scipy import signal
num = [1, 0, 0]
den = [1, 0.5, 0.25]
system = (num, den, 1)
t, y_dlsim = signal.dlsim(system, e)
print(f'y(dlsim) {y_dlsim[:3]}')
print(f' the experimental mean of y is {np.mean(y)}')

#======================================
# y(t) = 0.25e(t-1) + 0.5 e(t-2)
#======================================
num = [1, .25, 0.5]
den = [1, 0, 0]
system = (num, den, 1)
t, y_dlsim = signal.dlsim(system, e)
print(f'y(dlsim) {y_dlsim[:3]}')
print(f' the experimental mean of y is {np.mean(y_dlsim)}')