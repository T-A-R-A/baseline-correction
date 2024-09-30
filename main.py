#!/usr/bin/env python3

# %%
import bline_correction as bc 
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science','nature', 'notebook', 'grid'])

# %%
np.random.seed(42)
x = np.linspace(0, 10, 1000)
y_true = 2 * np.sin(2 * x) + 3 * np.cos(3 * x)
baseline = 5 + 0.5 * x + 0.1 * x**2
y_noisy = y_true + baseline + np.random.normal(0, 0.5, x.shape)


# %%
# Apply baseline correction
baseline_est = bc.adaptive_arpls(y_noisy)

# Plot results
plt.figure(figsize=(10, 10))
plt.plot(x, y_noisy, label='Original Signal')
plt.plot(x, baseline_est, label='Estimated Baseline')
plt.plot(x, y_noisy - baseline_est, label='Corrected Signal')
plt.legend()
plt.title('Adaptive ARPLS Baseline Correction')
plt.xlabel('X')
plt.ylabel("Signal")
plt.show()



