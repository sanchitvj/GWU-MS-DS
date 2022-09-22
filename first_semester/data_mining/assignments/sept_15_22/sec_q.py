import numpy as np

def get_max(arr1, arr2):
    m = np.maximum(arr1, arr2)
    return m

if __name__ == "__main__":
        
    arr1 = [23, 98, 38, 71]
    arr2 = [13, 245, 91, 36]
    m = get_max(arr1, arr2)
    print(m)