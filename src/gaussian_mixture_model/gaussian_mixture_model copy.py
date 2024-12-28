import matplotlib.pyplot as plt
import numpy as np

path = "data/old_faithful.txt"

xs = np.loadtxt(path)

print(xs.shape)
print(xs[0])

plt.scatter(xs[:, 0], xs[:, 1])
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time (min)")

plt.show()
