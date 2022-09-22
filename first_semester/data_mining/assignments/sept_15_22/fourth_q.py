import numpy as np

rand_var = np.random.normal(2, 5, 100)

nan_occurances = 20

nan_index = np.random.choice(rand_var.size, nan_occurances)

rand_var.ravel()[nan_index] = np.nan

print(rand_var)