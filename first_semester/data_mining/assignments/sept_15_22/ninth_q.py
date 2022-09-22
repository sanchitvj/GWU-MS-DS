import numpy as np

pois_rand_var = np.random.poisson(10, 20)

print(f"Original array: {pois_rand_var}")
print(f"Sorted array: {sorted(pois_rand_var)}")