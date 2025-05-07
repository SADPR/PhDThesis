
import numpy as np
import matplotlib.pyplot as plt

# LaTeX-friendly plot settings
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]
})
plt.rc('font', size=13)

# Load displacement data
fom_tip_disp = np.load("tip_node_magnitude_vector_fom.npy")
rom_tip_disp = np.load("tip_node_magnitude_vector_rom.npy")

# Time array (assuming constant time step)
dt = 0.1
t_vals = np.arange(0, len(fom_tip_disp) * dt, dt)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t_vals, fom_tip_disp, label="FOM", color="black", linewidth=2)
plt.plot(t_vals, rom_tip_disp, label="ROM", color="royalblue", linestyle="--", linewidth=2)

plt.xlabel(r"Time $t$ [s]")
plt.ylabel(r"Tip displacement magnitude [m]")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("tip_displacement_comparison.pdf", dpi=300, bbox_inches='tight')
plt.show()

