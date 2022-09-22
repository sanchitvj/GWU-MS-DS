import numpy as np

rand_var = np.random.normal(2, 5, 100)

mean = np.average(rand_var)
var = np.var(rand_var)
std = np.std(rand_var)

print(f"Mean: {mean:0.3f}")
print(f"Variance: {var:0.3f}")
print(f"Standard Deviation: {std:0.3f}")