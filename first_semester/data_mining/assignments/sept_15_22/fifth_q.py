import numpy as np

rand_var = np.random.normal(2, 5, 100)

nan_occurances = np.random.randint(1, 20) # inserting random number of NaN 
nan_index = np.random.choice(rand_var.size, nan_occurances)
rand_var.ravel()[nan_index] = np.nan

num = np.sum(np.isnan(rand_var))
print(num)