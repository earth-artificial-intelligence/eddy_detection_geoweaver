# setting the vmin and vmax using the eddy 'plot_variable' method
from eddy_paths import *
from copy import deepcopy
from matplotlib import pyplot as plt

ax, m = plot_variable(
    g,
    "adt",
    f"ADT (m) before high-pass filter",
    vmin=-0.15,
    vmax=0.15,
)
plt.savefig('/Users/lakshmichetana/ML_eddies_Output/ADT(m)_before_high-pass_filter.png', bbox_inches ="tight")
wavelength_km = 700

g_filtered = deepcopy(g)

g_filtered.bessel_high_filter("adt", wavelength_km)
ax, m = plot_variable(
    g_filtered,
    "adt",
    f"ADT (m) filtered (Final: {wavelength_km} km)",
    vmin=-0.15,
    vmax=0.15,
)

plt.savefig('/Users/lakshmichetana/ML_eddies_Output/ADT(m)-filtered.png', bbox_inches ="tight")

