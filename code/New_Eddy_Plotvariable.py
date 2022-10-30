# setting the vmin and vmax using the eddy 'plot_variable' method
from eddy_paths import *
from copy import deepcopy
from matplotlib import pyplot as plt

#updated the vmin and vmax to -1 and 1
ax, m = plot_variable(
    g,
    "adt",
    f"ADT (m) before high-pass filter",
    vmin=-5,
    vmax=5,
)
plt.savefig('/Users/lakshmichetana/ML_eddies_Output/ADT(m)_before_high-pass_filter_with_updatedVminVmax&Wavelength_KM.png', bbox_inches ="tight")
#updated wavelength covered kilometers to 100 from 700
wavelength_km = 100

g_filtered = deepcopy(g)

g_filtered.bessel_high_filter("adt", wavelength_km)
ax, m = plot_variable(
    g_filtered,
    "adt",
    f"ADT (m) filtered (Final: {wavelength_km} km)",
    vmin=-5,
    vmax=5,
)

plt.savefig('/Users/lakshmichetana/ML_eddies_Output/ADT(m)-filtered_with_updatedVminVmax&Wavelength_KM.png', bbox_inches ="tight")

