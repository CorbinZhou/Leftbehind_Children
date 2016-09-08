import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('GAdata1.txt')

plt.plot(results['generation'],results['train'])
plt.plot(results['generation'],results['validation'])
# plt.plot(results['generation'],results['test'])
plt.show()