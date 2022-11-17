# setting the vmin and vmax using the eddy 'plot_variable' method
#from eddy_paths import *
from eddy_paths import figOutputFolder, plot_variable, g
from copy import deepcopy
from matplotlib import pyplot as plt

#updated the vmin and vmax to -1 and 1
ax, m = plot_variable(
    g,
    "adt",
    f"ADT (m) before high-pass filter",
    vmin=-1,
    vmax=1,
)
plt.savefig(f'{figOutputFolder}/ADT(m)_before_high-pass_filter.png', bbox_inches ="tight")
#updated wavelength covered kilometers to 500 from 700
wavelength_km = 500

g_filtered = deepcopy(g)

g_filtered.bessel_high_filter("adt", wavelength_km)
ax, m = plot_variable(
    g_filtered,
    "adt",
    f"ADT (m) filtered (Final: {wavelength_km} km)",
    vmin=-1,
    vmax=1,
)

plt.savefig(f'{figOutputFolder}/ADT(m)-filtered.png', bbox_inches ="tight")

