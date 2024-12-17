import numpy as np
import matplotlib.pyplot as plt
from sodshock import *
import timeit

cell_counts = np.arange(2, 401, 1)
times = np.zeros_like(cell_counts, dtype=np.float64)

for i in np.arange(cell_counts.size):
    print(f"Cells: {cell_counts[i]}")
    times[i] = timeit.timeit(f"sodshock.sod_shock(cells = {cell_counts[i]}, x_start = 0., x_end = 1., t_final = 0.4, t_steps = 40)", 
                             setup="import sodshock", number=1)
    print(times[i])

# print(times)
plt.plot(cell_counts, times)
plt.title("Time Taken vs. Number of Cells Used")
plt.xlabel("Number of Cells")
plt.ylabel("Time Taken By Simulation (s)")
# plt.savefig("../Figures/lo_complexity.png", dpi=400)
plt.show()