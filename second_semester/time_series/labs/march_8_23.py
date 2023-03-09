
import numpy as np
from scipy import signal

np.random.seed(6313)
mean = 0
std = 1
N = 1000
e = np.random.normal(mean, std, N)

# method 1
y = np.zeros(len(e))
for i in range(len(e)):
    if i == 0:
        y[0] = e[0]
    elif i == 1:
        y[i] = -0.5 * y[i-1] + e[i]
    else:
        y[i] = -0.5 * y[i-1] + e[i] - 0.25 * y[i-2]

print(y[:3])


num = [1, 0, 0]
den = [1, 0.5, 0.25]
system = (num, den, 1)
t, y_dlsim = signal.dlsim(system, e)
print(y_dlsim[:3])

print("experimental mean of y is: ", np.mean(y))

