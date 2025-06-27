import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("ld_N_10_beamw_???_loadtest_Xv.txt")

plt.plot(data[:,0], data[:,1])

plt.xscale('log')
plt.yscale('log')

plt.show()
