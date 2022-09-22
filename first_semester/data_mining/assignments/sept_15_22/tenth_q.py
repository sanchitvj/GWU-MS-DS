import numpy as np

pois_rand_var = np.random.poisson(10, 20)

min_max_norm = (pois_rand_var - np.min(pois_rand_var)) / (np.max(pois_rand_var) - np.min(pois_rand_var))

print("Min max noramlized array: ", min_max_norm)