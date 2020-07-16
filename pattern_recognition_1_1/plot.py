import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV spikes
output_data = np.loadtxt("output.csv", delimiter=",",
                         dtype={"names": ("time", "y1", "y2", "y3", "y_star1", "y_star2", "y_star3"),
                                "formats": (np.float, np.float, np.float, np.float, np.float, np.float, np.float)})

# Create plot
figure, axes = plt.subplots(3, sharex=True)

# Plot Y*
axes[0].plot(output_data["time"], output_data["y1"], color="blue")
axes[0].plot(output_data["time"], output_data["y_star1"], color="blue", linestyle="--", alpha=0.5)
axes[1].plot(output_data["time"], output_data["y2"], color="blue")
axes[1].plot(output_data["time"], output_data["y_star2"], color="blue", linestyle="--", alpha=0.5)
axes[2].plot(output_data["time"], output_data["y2"], color="blue")
axes[2].plot(output_data["time"], output_data["y_star3"], color="blue", linestyle="--", alpha=0.5)

for t in range(0, 5000, 1000):
    for a in axes:
        a.axvline(t, linestyle="--", color="gray")
# Show plot
plt.show()
