# Process data to generate ground truth using py-eddy-tracker

from dependency import *
from plot_utils import *
from matplotlib.path import Path
from py_eddy_tracker.poly import create_vertice

def generate_segmentation_mask_from_file(
    gridded_ssh_file,
    date,
    ssh_var="adt",
    u_var="ugosa",
    v_var="vgosa",
    high_pass_wavelength_km=700,
    x_offset=0,
    y_offset=0,
):
    g, g_filtered, anticyclonic, cyclonic = identify_eddies(
        gridded_ssh_file, date, ssh_var, u_var, v_var, high_pass_wavelength_km
    )
    mask = generate_segmentation_mask(
        g_filtered, anticyclonic, cyclonic, x_offset, y_offset
    )
    var = g.grid(ssh_var)
    var_filtered = g_filtered.grid(ssh_var)
    return var, var_filtered, mask


def identify_eddies(
    gridded_ssh_file,
    date,
    ssh_var="adt",
    u_var="ugosa",
    v_var="vgosa",
    high_pass_wavelength_km=700,
):
    g = RegularGridDataset(gridded_ssh_file, "longitude", "latitude")
    g_filtered = deepcopy(g)  # make a copy so we don't alter the original
    g_filtered.bessel_high_filter(ssh_var, high_pass_wavelength_km)
    anticyclonic, cyclonic = g_filtered.eddy_identification(ssh_var, u_var, v_var, date)
    return g, g_filtered, anticyclonic, cyclonic


def generate_segmentation_mask(
    grid_dataset, anticyclonic, cyclonic, x_offset, y_offset, plot=False
):
    """
    Creates a numpy array to store the segmentation mask for the grid_dataset.
    The mask contains classes 0: no eddy, 1: anticyclonic eddy, 2: cyclonic eddy.
    """
    assert (
        cyclonic.sign_legend == "Cyclonic"
        and anticyclonic.sign_legend == "Anticyclonic"
    ), "Check whether the correct order for (anti)cyclonic observations were provided."
    mask = np.zeros(grid_dataset.grid("adt").shape, dtype=np.uint8)
    # cyclonic should have the same: x_name = 'contour_lon_e', y_name = 'contour_lat_e'
    x_name, y_name = anticyclonic.intern(False)
    for eddy in anticyclonic:
        x_list = (eddy[x_name] - x_offset) % 360 + x_offset
        vertices = create_vertice(x_list, eddy[y_name] + y_offset)
        i, j = Path(vertices).pixels_in(grid_dataset)
        mask[i, j] = 1

    for eddy in cyclonic:
        x_list = (eddy[x_name] - x_offset) % 360 + x_offset
        y_list = eddy[y_name] + y_offset
        i, j = Path(create_vertice(x_list, y_list)).pixels_in(grid_dataset)
        mask[i, j] = 2

    if plot:
        ax, m,fig = plot_variable(grid_dataset, mask, "Segmentation Mask", cmap="viridis")
    return mask
