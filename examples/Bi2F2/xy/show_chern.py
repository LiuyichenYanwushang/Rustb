import numpy as np
import matplotlib.pyplot as plt

C=np.loadtxt("Chern.txt")
C=C[::-1]

plt.imshow(C)
plt.colorbar()
plt.show()
