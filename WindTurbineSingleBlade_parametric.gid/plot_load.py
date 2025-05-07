import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]
})
plt.rc('font', size=13)


# Force pulse definition
def compute_pressure_time_series(pulse_schedule, pulse_duration, t_end, dt=0.01):
    t_vals = np.arange(0.0, t_end + dt, dt)
    pressure_vals = np.zeros_like(t_vals)

    for i, t in enumerate(t_vals):
        for t_start, amplitude in pulse_schedule:
            t_end_pulse = t_start + pulse_duration
            if t_start <= t < t_end_pulse:
                t_local = t - t_start
                pressure_vals[i] = amplitude * np.sin(np.pi * t_local / pulse_duration)**2
                break

    return t_vals, pressure_vals

# Geometry
L = 43.2
x = L
pulse_duration = 2.0
t_end = 8.0
dt = 0.01

# Amplitudes in Pascals
pulse_schedule_train = [(0.0, 5e4), (2.0, 4e4), (4.0, 6e4), (6.0, 3e4)]
pulse_schedule_test  = [(0.0, 5e4), (2.0, 3.5e4), (4.0, 6.5e4), (6.0, 2.5e4)]

# Compute
t_vals, p_train = compute_pressure_time_series(pulse_schedule_train, pulse_duration, t_end, dt)
_, p_test = compute_pressure_time_series(pulse_schedule_test, pulse_duration, t_end, dt)

# At blade tip
p_train_tip = p_train * (x / L)**2
p_test_tip = p_test * (x / L)**2

# Plot in kPa
plt.figure(figsize=(8, 4))
plt.plot(t_vals, p_train_tip / 1e3, label="Training input", color="royalblue", linewidth=2)
plt.plot(t_vals, p_test_tip / 1e3, label="Test input", color="darkorange", linestyle="--", linewidth=2)

plt.xlabel("Time t [s]")
plt.ylabel("Pressure at tip [kPa]")
plt.title("Training and Test Pressure Inputs at Blade Tip")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("training_vs_test_pressure_tip_kpa.pdf", dpi=300, bbox_inches="tight")
plt.show()


