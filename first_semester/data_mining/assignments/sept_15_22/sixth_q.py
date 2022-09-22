import numpy as np

rand_var = np.random.normal(2, 5, 100)

nan_occurances = np.random.randint(1, 20) # inserting random number of NaN 
nan_index = np.random.choice(rand_var.size, nan_occurances)
rand_var.ravel()[nan_index] = np.nan

reshaped_rand_var = rand_var.reshape((10, 10))

print(reshaped_rand_var)