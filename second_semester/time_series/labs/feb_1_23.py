import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import autocorr

y = [1,2,3,4,5]
lags = 5
title = 'dummy'
ryy = autocorr(y, lags, title)

