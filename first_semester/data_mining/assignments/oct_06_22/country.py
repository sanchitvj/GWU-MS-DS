
from re import X
import pandas as pd


with open('countries.txt', 'r') as f:
    t = f.read()

c = []
for i in t.split("  "):
    # if len(i) > 5:
    x = i.replace("\n", "").replace('"','')
    if len(x) > 2:
        c.append(x)

# c = [str(j) for j in c]
print(c)
# new_t = ",".join(t)
# print(new_t)
# new_t = [int(num) for num in t.split("  ")]

# print(t.split("  ").replace('"',''))