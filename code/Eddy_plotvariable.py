# Write first python in Geoweaver
from copy import deepcopy

ax, m = plot_variable(
    g,
    "adt",
    f"ADT (m) before high-pass filter",
    vmin=-0.15,
    vmax=0.15,
)
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
