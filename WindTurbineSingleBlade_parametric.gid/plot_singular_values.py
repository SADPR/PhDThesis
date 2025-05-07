import numpy as np
import matplotlib.pyplot as plt

# LaTeX-friendly plot settings
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]
})
plt.rc('font', size=13)

# Load singular values
singular_values = np.load("rom_data_0_tol/SingularValuesVector.npy")
j_vals = np.arange(1, len(singular_values) + 1)

# Compute linear and squared cumulative energy loss
total_sum = np.sum(singular_values)
cumulative_sum = np.cumsum(singular_values)
linear_relative_loss = 1.0 - (cumulative_sum / total_sum)

squared_total_sum = np.sum(singular_values**2)
squared_cumulative_sum = np.cumsum(singular_values**2)
squared_relative_loss = 1.0 - (squared_cumulative_sum / squared_total_sum)

# Plot 1: Linear energy loss
plt.figure(figsize=(8, 5))
plt.plot(j_vals, linear_relative_loss, linewidth=2, color="darkorange", label=r"$1 - \sum_{i=1}^{n} \sigma_i \,/\, \sum_{i=1}^{m} \sigma_i$")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel(r"Singular value index $n$")
plt.ylabel(r"$1 - \frac{\sum_{i=1}^{n} \sigma_i}{\sum_{i=1}^{m} \sigma_i}$")
# plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("svd_relative_loss_linear.pdf", dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Squared energy loss
# plt.figure(figsize=(8, 5))
# plt.plot(j_vals, squared_relative_loss, linewidth=2, color="royalblue", label=r"$1 - \sum_{i=1}^{n} \sigma_i^2 \,/\, \sum_{i=1}^{m} \sigma_i^2$")
# plt.yscale("log")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.xlabel(r"Singular value index $n$")
# plt.ylabel(r"$1 - \frac{\sum_{i=1}^{n} \sigma_i^2}{\sum_{i=1}^{m} \sigma_i^2}$")
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig("svd_relative_loss_squared.pdf", dpi=300, bbox_inches='tight')
# plt.show()


