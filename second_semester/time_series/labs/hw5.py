import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import ar_ma_dlsim, sm_arma_process
from scipy import signal

seed_num = 6313
np.random.seed(seed_num)

# Example 2: ARMA (0,1):   y(t) =   e(t) + 0.5e(t-1)
# Example 3: ARMA (1,1):   y(t) + 0.5y(t-1) = e(t) + 0.25e(t-1)            x out of CI est coeffs
# Example 4: ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
# Example 5: ARMA (2,1):  y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)  x out of CI est coeffs
# Example 6: ARMA (1,2):  y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)  x wrong est coeffs
# Example 7: ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)               x wrong est coeffs
# Example 8: ARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)

# N = int(input("Enter number of samples (N): "))
# na = int(input("Enter AR order: "))
# nb = int(input("Enter MA order: "))
#
# den, num = [], []
# print("Enter the coefficients of AR (separated by space):")
# den = [float(x) for x in input().split()]
# print("Enter the coefficients of MA (separated by space):")
# num = [float(x) for x in input().split()]

N = 1000
num, den = [1, 0.5], [1, 0.25]
na, nb = 1, 1
e = np.random.normal(0, 1, N)
y = ar_ma_dlsim(e, num, den, "ar")

true_theta = [i for i in den[1:na + 1]] + [i for i in num[1:nb + 1]]
# true_theta = np.array([den[1], den[2], num[1]])
print("True coefficients: ", true_theta)
true_theta = np.array(true_theta)


def arma_model(theta, y, e, na, nb):
    np.random.seed(seed_num)
    T = len(y)
    n = max(na, nb)
    # y_pred = np.zeros(T)
    e_hat = np.zeros(T)
    # num, den = np.array(n+1) + np.array(n+1)
    num, den = [1.0] + theta[n:], [1.0] + theta[:n]
    num = num if len(num) == n + 1 else num + [0.0] * (n + 1 - len(num))  # [i for i in np.zeros(n+1-len(num))]
    den = den if len(den) == n + 1 else den + [0.0] * (n + 1 - len(den))  # [i for i in np.zeros(n+1-len(den))]

    assert len(num) == len(den), f"Length of num {num} must be equal to length of den {den}."
    # system = (num, den, 1)
    # _, e_hat = signal.dlsim(system, y)
    # e_hat = y_pred
    y_pred = ar_ma_dlsim(e, num, den, "ar")
    for t in range(0, T):
        # y_pred[t] = -theta[0] * y[t - 1] - theta[1] * y[t - 2] + theta[2] * e[t - 1] + e[t]
        e_hat[t] = y[t] - y_pred[t]
    # print(e_hat.shape)
    return y_pred, e_hat  # .reshape(N,)#.flatten()  #


# Define the Levenberg-Marquardt algorithm
def levenberg_marquardt(y, theta0, e, mu0=0.01, mu_max=1e10, epsilon=1e-3, h=1e-6, max_iter=100):
    np.random.seed(seed_num)
    N = len(y)
    n = len(theta0)
    theta = theta0
    mu = mu0
    iterations = 0
    sse_new, iters = [], []
    while iterations < max_iter:
        iterations += 1
        y_pred, e_hat = arma_model(theta, y, e, na, nb)
        J = np.zeros((N, n))
        for i in range(n):
            params_perturb = theta  # np.array(params)
            params_perturb[i] += h
            _, residuals_perturb = arma_model(params_perturb, y, e, na, nb)  # model_func(params_perturb, data)
            # print(np.array(residuals_perturb).reshape(10000,).shape)
            J[:, i] = (e_hat - residuals_perturb) / h

        A = J.T.dot(J)
        # A_plus_mui = A + mu * np.identity(n)
        # delta_theta = np.linalg.inv(A_plus_mui).dot(J.T.dot(e_hat))  # np.linalg.solve(A, J.T @ e_hat)
        A += mu * np.identity(n)
        delta_theta = np.linalg.inv(A).dot(J.T.dot(e_hat))
        theta_new = theta + delta_theta
        y_pred_new, e_hat_new = arma_model(theta_new, y, e, na, nb)
        SSE_new = np.sum(e_hat_new ** 2)
        SSE = np.sum(e_hat ** 2)
        # SSE_new = e_hat_new.T.dot(e_hat_new)  # .reshape(N, 1)
        # SSE = e_hat.T.dot(e_hat)
        sse_new.append(SSE_new)
        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon:
                sigma_e_squared_hat = SSE_new / (N - n)
                cov_theta_hat = sigma_e_squared_hat * np.linalg.inv(A)
                print(f"Converged in {iterations} iterations.")
                iters.append(iterations)
                return theta_new, sigma_e_squared_hat, cov_theta_hat, sse_new, iters
            else:
                theta = theta_new
                mu /= 10
        else:
            mu *= 10
            if mu > mu_max:
                print("Failed to converge. Maximum mu reached.")
                return theta, None, None, None, None

    print("Failed to converge. Maximum iterations reached.")
    return theta, None, None, None, None


theta0 = np.zeros(len(true_theta), dtype=np.float64)
theta_hat, sigma_e_squared_hat, cov_theta_hat, sse_new, iters = levenberg_marquardt(y, theta0, e)

# plt.plot(sse_new)
# plt.title("SSE vs. Iterations")
# plt.xlabel("Iterations")
# plt.ylabel("SSE")
# plt.show()

zeros = np.roots(np.r_[1, theta_hat[:na]])
poles = np.roots(np.r_[1, theta_hat[na:]])
print("Zeros: ", zeros)
print("Poles: ", poles)

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

import statsmodels.api as sm

arma_data, ryt, na, nb = sm_arma_process(15)

# Estimate ARMA model coefficients from the generated data
model = sm.tsa.ARIMA(arma_data, order=(na, 0, nb))  # ARIMA model with specified order
results = model.fit()  # method_kwargs={'warn_convergence': False})  # Fit the ARIMA model
estimated_coefficients = results.arparams  # Get the estimated AR coefficients
print(results.summary())
# Display the estimated coefficient(s) on the console
print("Estimated AR Coefficient(s): ", estimated_coefficients)
# for i in range(na):
#     print(f"AR{i + 1}: {estimated_coefficients[i]:.3f}")
