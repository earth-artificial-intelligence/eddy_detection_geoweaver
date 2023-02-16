# sea surface height (SSH preprocessing)


from dependency import *
from plot_utils import plot_variable, save_fig_and_relesase_memory
from data_loader import example_file


date = datetime(2019, 1, 1)
g = RegularGridDataset(example_file, "longitude", "latitude")

ax, m, fig = plot_variable(
    g,
    "adt",
    f"ADT (m) before high-pass filter",
    vmin=-0.15,
    vmax=0.15,
)

save_fig_and_relesase_memory(ax, m, fig)

wavelength_km = 700
g_filtered = deepcopy(g)
g_filtered.bessel_high_filter("adt", wavelength_km)

ax, m, fig = plot_variable(
    g_filtered,
    "adt",
    f"ADT (m) filtered (Final: {wavelength_km} km)",
    vmin=-0.15,
    vmax=0.15,
)

save_fig_and_relesase_memory(ax, m, fig)


