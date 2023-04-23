import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import ar_ma_dlsim, sm_arma_process
from scipy import signal

seed_num = 6313
np.random.seed(seed_num)

# %% Final
def residual(y, theta, p, q):
    np.random.seed(6313)

    num = [theta[i] for i in range(p)]
    den = [theta[p+j] for j in range(q)]
    num = np.r_[1, num]
    den = np.r_[1, den]

    if len(den) >= len(num):
        num = np.pad(num, (0, len(den) - len(num)))
    if len(den) < len(num):
        den = np.pad(den, (0, len(num) - len(den)))

    system = (num, den, 1)
    _, res = signal.dlsim(system, y)

    return res.flatten()


# %% Final
def jacobian(theta,y, e, p, q, delta):
    np.random.seed(6313)
    res = residual(y, theta, p, q)
    num_samples = len(y)
    X = np.zeros((num_samples, p + q))

    for i in range(p + q):
        theta_ = theta.copy()
        theta_[i] += delta
        e_new = residual(y, theta_, p, q)
        X[:, i] = (res - e_new) / delta
        # for i in range(p):
        #     X[t, i] = -y[t - i - 1]
        # for j in range(q):
        #     X[t, p + j] = -e[t - j - 1]

    return X


# %% Final
def LM_ARMA(y, p, q, theta_init=None, delta=1e-6, max_iter=10, mu=0.01, mu_max=10e10, epsilon=1e-3):
    np.random.seed(6313)
    theta = theta_init
    if theta_init is None:
        theta = np.zeros(p + q)

    num_iter = 0

    while num_iter < max_iter:
        # Step 1
        e = residual(y, theta, p, q)
        SSE = e.T @ e
        X = jacobian(theta, y, e, p, q, delta)
        A = X.T @ X
        I = np.identity(p + q)
        g = X.T @ e

        # Step 2
        # delta_theta = np.linalg.solve(A + mu * I, g)
        delta_theta = np.linalg.inv(A + (mu * I)) @ g
        theta_new = theta + delta_theta

        e_new = residual(y, theta_new, p, q)
        SSE_new = e_new.T @ e_new

        # Step 3
        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon:
                theta_hat = theta_new
                sigma_e_squared = SSE_new / (len(y) - p - q)
                cov_theta_hat = sigma_e_squared * np.linalg.inv(A)
                return theta_hat, sigma_e_squared, cov_theta_hat, num_iter
            else:
                theta = theta_new.copy()
                mu /= 10
        while SSE_new >= SSE:
            mu *= 10
            if mu > mu_max:
                print("Results:", theta)
                print("Error: mu exceeded mu_max")
                return theta, None, None, num_iter

            delta_theta = np.linalg.inv(A + (mu * I)) @ g
            theta_new = theta + delta_theta

            e_new = residual(y, theta_new, p, q)
            SSE_new = e_new.T @ e_new

        num_iter += 1
        theta = theta_new.copy()

    print("Results:", theta_new)
    print("Error: Maximum number of iterations reached")
    return theta_new, None, None, num_iter


N = 1000
num, den = [0.5, -0.4], [0.5, 0.2]
na, nb = 2, 2
e = np.random.normal(0, 1, N)
y = ar_ma_dlsim(e, num, den, "ar")

# true_theta = [i for i in den[1:na + 1]] + [i for i in num[1:nb + 1]]
# true_theta = np.array([den[1], den[2], num[1]])
true_theta = den + num
print("True coefficients: ", true_theta)
true_theta = np.array(true_theta)

theta0 = np.zeros(len(true_theta), dtype=np.float64)
# print(true_theta)
# print(theta0)
theta_hat, sigma_e_squared_hat, cov_theta_hat, num_iter = LM_ARMA(y, na, nb, theta0)

if sigma_e_squared_hat is not None:
    for i in range(na + nb):
        process = "AR"
        if i >= na:
            process = "MA"
        print(f"Estimated {process} coeff{i + 1}: {(theta_hat[i]):.3f}")
        print(f"Confidence interval {process} coeff{i + 1}: ",
              [true_theta[i] - 2 * np.sqrt(cov_theta_hat[i, i]), true_theta[i] + 2 * np.sqrt(cov_theta_hat[i, i])])

    print("Covariance matrix of estimated coefficients:\n", cov_theta_hat)
    print(f"Estimated error variance: {sigma_e_squared_hat:.4f}")