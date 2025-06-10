from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib import cm

# Load CSV data passed as first argument
data = np.loadtxt(sys.argv[1], delimiter=',')

# Expected 3D shape dimensions
d1 = 128
d2 = 256
d3 = 64

# Threshold for difference
th = 0

# Reshape data: ignore first column in each row
faulty = (data[1, 1:]).reshape(d1, d2, d3)
free = (data[0, 1:]).reshape(d1, d2, d3)

# Compute absolute difference
diff_f = np.abs(free - faulty)

# Mask where the difference exceeds threshold
mask = diff_f > th
x = np.where(mask)[0]
y = np.where(mask)[1]
z = np.where(mask)[2]

# Choose color values based on user input
if sys.argv[2] == "faulty":
    colo = faulty[mask]
    volume = data[1, 1:]
    array3d = faulty
elif sys.argv[2] == "free":
    colo = free[mask]
    volume = data[0, 1:]
    array3d = free
else:
    raise ValueError("Second argument must be either 'faulty' or 'free'")

# Create plot
fig = plt.figure(figsize=(8.2, 7))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot with color mapping
img = ax.scatter(x, y, z, marker='o', s=6, c=colo, cmap='Greens')
plt.colorbar(img, ax=ax)

# Set axis limits
ax.set_xlim(0, d1)
ax.set_ylim(0, d2)
ax.set_zlim(0, d3)

# Extract width info from filename (second-to-last part if using underscores)
width = (sys.argv[1]).split("_")[-2]
title_prefix = "FAULTY" if sys.argv[2] == "faulty" else "FREE"
ax.set_title(f"{title_prefix}   width: {width}")

# Axis labels
ax.set_xlabel('dim 0')
ax.set_ylabel('dim 1')
ax.set_zlabel('dim 2')

# Verification 
start = 1308997
last = 1839549
def convert(i):
    flat_index = i - 1
    i = flat_index // (d2 * d3)
    rem = flat_index % (d2 * d3)
    j = rem // d3
    k = rem % d3

    flat_value = volume[flat_index]
    reshaped_value = array3d[i, j, k]

    #print(f"Flat index {flat_index} → (i={i}, j={j}, k={k})")
    #print(f"Flat value: {flat_value}")
    #print(f"Value at [{i}, {j}, {k}]: {reshaped_value}")
    print(f"Match? {flat_value == reshaped_value}")
    return [i, j, k]

# Plot the point for visual confirmation
ax.scatter(x[0], y[0], z[0], c='red', s=60, marker='x', label=f'Sample {start}')
ax.scatter(x[-1], y[-1], z[-1], c='blue', s=60, marker='x', label=f'Sample {last}')
print(f"Start: ({x[0]}, {y[0]}, {z[0]})")
print(f"End: ({x[-1]}, {y[-1]}, {z[-1]})")
ax.legend()

# Display plot
plt.show()

# 89974 - (0, 1, 1)
# 90367 - (127, 255, 62)
