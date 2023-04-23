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


def arma_model(theta, y, na, nb):
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
    e_hat = ar_ma_dlsim(y, den, num, "ar")
    # for t in range(0, T):
    #     # y_pred[t] = -theta[0] * y[t - 1] - theta[1] * y[t - 2] + theta[2] * e[t - 1] + e[t]
    #     e_hat[t] = y[t] - y_pred[t]
    # print(e_hat.shape)
    return e_hat.reshape(N,)


def step1(theta, y, e_hat, na, nb, h=1e-6):
    N = len(y)
    n = len(theta)
    J = np.zeros((N, n))
    for i in range(n):
        params_perturb = theta  # np.array(params)
        params_perturb[i] += h
        residuals_perturb = arma_model(params_perturb, y, na, nb)  # model_func(params_perturb, data)
        J[:, i] = (e_hat - residuals_perturb) / h

    # A = np.dot(J.T, J)
    # g = np.dot(J.T, e_hat)

    return J


def step2(theta, A, g, y, mu):
    n = len(theta)
    delta_theta = np.dot(np.linalg.inv(A + mu * np.identity(n)), g)
    theta_new = theta + delta_theta
    # _, e_hat_new = arma_model(theta_new, y, e, na, nb)
    # SSE_new = np.dot(e_hat_new.T, e_hat_new)

    return delta_theta, theta_new


def step3(theta, y, mu0=0.01, max_iter=100, epsilon=1e-3, mu_max=1e10):
    N, n = len(y), len(theta)
    mu = mu0
    iteration = 0
    sse_new = []
    while iteration < max_iter:
        # A, g, SSE = step1(theta, y, e, na, nb)
        # SSE_new, delta_theta, theta_new = step2(theta, A, g, y, mu)
        e_hat = arma_model(theta, y, na, nb)
        SSE = np.sum(e_hat ** 2)
        J = step1(theta, y, e_hat, na, nb)
        A = J.T.dot(J)
        g = J.T.dot(e_hat)
        delta_theta, theta_new = step2(theta, A, g, y, mu)

        e_hat_new = arma_model(theta_new, y, na, nb)
        SSE_new = np.sum(e_hat_new ** 2)
        sse_new.append(SSE_new)

        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon:# * np.linalg.norm(theta):
                theta_hat = theta_new
                sigma_e_squared_hat = SSE_new / (N - n)
                cov_theta_hat = sigma_e_squared_hat * np.linalg.inv(A)
                print(f"Converged in {iteration} iterations.")
                return theta_hat, sigma_e_squared_hat, cov_theta_hat, sse_new
            else:
                theta = theta_new
                mu = mu / 10

        while SSE_new >= SSE:
            mu *= 10
            if mu > mu_max:
                print("Failed to converge. Maximum mu reached.")
                return theta, None, None, None
            # A, g, SSE = step1(theta, y, e, na, nb)
            # SSE_new, delta_theta, theta_new = step2(theta, A, g, y, mu)
            # delta_theta, theta_new = step2(theta, A, g, y, mu)
            delta_theta = np.linalg.inv(A + (mu * np.identity(n))) @ g
            theta_new = theta + delta_theta
            e_hat_new = arma_model(theta_new, y, na, nb)
            SSE_new = np.sum(e_hat_new ** 2)
            sse_new.append(SSE_new)

        iteration += 1
        if iteration > max_iter:
            print("Failed to converge. Maximum iterations reached.")
            return theta, None, None, None


N = 5000
num, den = [1, 0.5, -0.4], [1, 0.5, 0.2]
na, nb = 2, 2
e = np.random.normal(0, 1, N)
y = ar_ma_dlsim(e, num, den, "ar")

true_theta = [i for i in den[1:na + 1]] + [i for i in num[1:nb + 1]]
# true_theta = np.array([den[1], den[2], num[1]])
print("True coefficients: ", true_theta)
true_theta = np.array(true_theta)

theta0 = np.zeros(len(true_theta), dtype=np.float64)
theta_hat, sigma_e_squared_hat, cov_theta_hat, sse_new = step3(theta0, y)

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


