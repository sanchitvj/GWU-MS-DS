import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import ar_ma_dlsim
from scipy.linalg import toeplitz

np.random.seed(6313)

# N = int(input("Enter number of samples (N): "))
# order = int(input("Enter order of AR process: "))
# e = np.random.normal(mean, std, N)
#
# num, den = [], []
# for i in range(order + 1):
#     n = 1 if i == 0 else float(input(f"Enter numerator (coeff) {i + 1}: "))
#     d = 1 if i == 0 else float(input(f"Enter denominator (coeff) {i + 1}: "))
#     num.append(n), den.append(d)
#
N = 5000
num, den = [1, -0.5], [1, 0.5, 0.2]
system = (num, den, 1)
e = np.random.normal(0, 1, N)
# t, y_dlsim = signal.dlsim(system, e)
y_dlsim = ar_ma_dlsim(e, num, den, "ar")


# Generate ARMA(1,1) data
# T = 100000
true_theta = np.array([den[1], den[2], num[1]])
# y = np.zeros(T)
# e = np.random.normal(0, 1, T)
# for t in range(1, T):
#     y[t] = true_theta[0] * y[t - 1] + true_theta[1] * e[t - 1] + e[t]
y = y_dlsim

# Define the ARMA(1,1) model
def arma_model(theta, y, e):
    # np.random.seed(6313)
    T = len(y)
    y_pred = np.zeros(T)
    e_hat = np.zeros(T)
    # num, den = [1, theta[1]], [1, theta[0]]
    # system = (num, den, 1)
    # e = np.random.normal(0, 1, N)
    # y_dlsim = ar_ma_dlsim(e, num, den, "ar")
    for t in range(0, T):
        y_pred[t] = -theta[0] * y[t - 1] - theta[1] * y[t - 2] + theta[2] * e[t - 1] + e[t]
        e_hat[t] = y[t] - y_pred[t]
        # e_hat[t] = y[t] - y_dlsim[t]
    return y_pred, e_hat


# Define the Levenberg-Marquardt algorithm
def levenberg_marquardt(y, theta0, e, mu0=0.01, mu_max=1e10, epsilon=1e-3, h=1e-6, max_iter=100):
    # np.random.seed(6313)
    N = len(y)
    n = len(theta0)
    theta = theta0
    mu = mu0
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        y_pred, e_hat = arma_model(theta, y, e)
        J = np.zeros((N, n))
        # for t in range(1, N):
        #     J[t, 0] = y[t - 1]
        #     J[t, 1] = e_hat[t - 1]
        # J[0, 0] = 0
        # J[0, 1] = 0
        for i in range(n):
            params_perturb = theta  # np.array(params)
            params_perturb[i] += h
            # params_perturb[i] = params_perturb[i] + 1e-6
            # print(params_perturb[i])
            _, residuals_perturb = arma_model(params_perturb, y, e)  # model_func(params_perturb, data)
            J[:, i] = (e_hat - residuals_perturb) / h

        A = J.T.dot(J)
        A += mu * np.identity(n)
        delta_theta = np.linalg.inv(A).dot(J.T.dot(e_hat))  # np.linalg.solve(A, J.T @ e_hat)
        theta_new = theta + delta_theta
        y_pred_new, e_hat_new = arma_model(theta_new, y, e)
        SSE_new = np.sum(e_hat_new ** 2)
        SSE = np.sum(e_hat ** 2)
        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon * np.linalg.norm(theta):
                sigma_e_squared_hat = SSE_new / (N - n)
                cov_theta_hat = sigma_e_squared_hat * np.linalg.inv(A)
                return theta_new, sigma_e_squared_hat, cov_theta_hat
            else:
                theta = theta_new
                mu /= 10
        else:
            mu *= 10
            if mu > mu_max:
                print("Failed to converge. Maximum mu reached.")
                return theta, None, None

    print("Failed to converge. Maximum iterations reached.")
    return theta, None, None


# Initial guess for ARMA(1,1) coefficients
theta0 = np.array([0, 0, 0], dtype=np.float64)

# Estimate ARMA(1,1) coefficients using Levenberg-Marquardt algorithm
theta_hat, sigma_e_squared_hat, cov_theta_hat = levenberg_marquardt(y, theta0, e)

# Print results
print("Estimated AR coefficient: {:.3f}".format(-theta_hat[0]))
print("Estimated MA coefficient: {:.3f}".format(theta_hat[1]))
print("Estimated MA coefficient: {:.3f}".format(theta_hat[2]))

if sigma_e_squared_hat is not None:
    print(f"Estimated error variance: {sigma_e_squared_hat:.4f}")
    print("Confidence interval AR",
          [true_theta[0] - 2 * np.sqrt(cov_theta_hat[0, 0]), true_theta[0] + 2 * np.sqrt(cov_theta_hat[0, 0])])
    print("Confidence interval MA",
          [true_theta[1] - 2 * np.sqrt(cov_theta_hat[1, 1]), true_theta[1] + 2 * np.sqrt(cov_theta_hat[1, 1])])
    print("Covariance matrix of estimated coefficients:\n", cov_theta_hat)
