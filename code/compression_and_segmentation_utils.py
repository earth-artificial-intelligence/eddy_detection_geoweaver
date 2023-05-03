# Generate ground truth on a global scale helper functions

import multiprocessing
from dependency import *
from matplotlib.path import Path
from py_eddy_tracker.poly import create_vertice
from datetime import datetime
import os.path
from dependency import plt



# Helper for Ploting graphs

def start_axes(title):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
    ax.set_aspect("equal")
    ax.set_title(title, weight="bold")
    return ax, fig


def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.94, 0.05, 0.01, 0.9]))




def plot_variable(grid_object, var_name, ax_title, **kwargs):
    ax,fig = start_axes(ax_title)
    m = grid_object.display(ax, var_name, **kwargs)
    update_axes(ax, m)
    ax.set_xlim(grid_object.x_c.min(), grid_object.x_c.max())
    ax.set_ylim(grid_object.y_c.min(), grid_object.y_c.max())
    return ax, m, fig

def save_fig_and_relesase_memory(ax, m, fig):
    # TODO: change the function to account for a relevant name
    #fig.savefig( os.path.join("/home/chetana/ML_eddies/plots/", "0.png"))
    ax.cla()
    plt.close('all')


# Helpers to generate ground truth

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



# Compress segmask to npz files

def generate_masks_in_parallel(
    files,
    dates,
    ssh_var="adt",
    u_var="ugosa",
    v_var="vgosa",
    high_pass_wavelength_km=700,
    x_offset=-180,
    y_offset=0,
    num_processes=8,
    plot=False,
    save=True,
    test=False,
):
    args = [
        (file, date, ssh_var, u_var, v_var, high_pass_wavelength_km, x_offset, y_offset)
        for file, date in zip(files, dates)
    ]
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.starmap(generate_segmentation_mask_from_file, args)

    vars_ = []
    vars_filtered = []
    masks = []
    for result in results:
        vars_.append(result[0])
        vars_filtered.append(result[1])
        masks.append(result[2])

    # concatenate list into single numpy array and return
    masks = np.stack(masks, axis=0)
    vars_ = np.stack(vars_, axis=0).astype(np.float32)
    vars_filtered = np.stack(vars_filtered, axis=0).astype(np.float32)

    if save:
        # find common folder across all files
        common_folder = os.path.commonpath(files)
        if test is True:
            common_folder = "/home/chetana/ML_eddies/cds_ssh_2019_10day_interval"
        years = sorted(set([date.year for date in dates]))
        year_str = f"{years[0]}" if len(years) == 1 else f"{min(years)}-{max(years)}"
        save_path = os.path.join(
            common_folder, f"global_pet_masks_with_{ssh_var}_{year_str}.npz"
        )
        np.savez_compressed(
            save_path,
            masks=masks,
            dates=dates,
            var=vars_,
            var_filtered=vars_filtered,
        )
        print(f"Saved masks to {save_path}")

    return vars_, vars_filtered, masks


from itertools import product


def get_dates_and_files(years, months, days, folder, file_pattern):
    """
    Given a filename pattern and a list of years months and days,
    fill in the filename pattern with the date and return
    a list of filenames and a list of associated `datetime` objects.

    Args:
        years (list): list of years, e.g., [1993, 1994, 1995, 1996]
        months (list): list of months, e.g., [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        days (list): list of days, e.g., [1, 10, 20, 30] for every 10th day
        folder (str): folder where the files are located
        file_pattern (str): filename pattern, e.g.,
            "dt_global_twosat_phy_l4_{year:04d}{month:02d}{day:02d}_vDT2021.nc"

    Returns:
        files (list): full/absolute path to each netCDF file in the list of dates
        dates (list): list of `datetime` objects formed from the combination of years, months and days
    """
    dates, files = [], []
    for y, m, d in product(years, months, days):  # cartesian product
        try:
            date = datetime(y, m, d)
            file = os.path.join(folder, file_pattern.format(year=y, month=m, day=d))
            dates.append(date)
            files.append(file)
        # catch ValueError thrown by datetime if date is not valid
        except ValueError:
            pass
    years = f"{years[0]}" if len(years) == 1 else f"{min(years)}-{max(years)}"
    print(f"Found {len(dates)} files for {years}.")
    return dates, files


# Generate segmentation

def subset_arrays(
    masks,
    var,
    var_filtered,
    dates,
    lon_range,
    lat_range,
    resolution_deg,
    plot=False,
    ssh_var="adt",
    save_folder=None,
):
    """
    Subset the arrays to the given lon_range and lat_range.

    Args:
        masks (np.ndarray): Global eddy segmentation masks.
            Can be masks from multiple dates concatenated into one array
        var (np.ndarray): Global SSH value
        var_filtered (np.ndarray): Global SSH value after high-pass filter
        dates (list): List of `datetime` objects
        lon_range (tuple): Longitude range to subset to (lon_start, lon_end)
        lat_range (tuple): Latitude range to subset to (lat_start, lat_end)
        resolution_deg (float): Resolution of the SSH map in degrees
        plot (bool): Whether to plot a sample of the subsetted arrays
        ssh_var (str): SSH variable name. Defaults to "adt". Only used if save_folder is not None.
        save_folder (str): Folder to save the subsetted arrays to. Defaults to None.
            If None, the arrays are not saved.

    Returns:
        mask_subset (np.ndarray): Subsetted masks
        var_subset (np.ndarray): Subsetted var
        var_filtered_subset (np.ndarray): Subsetted var_filtered
        lon_subset (np.ndarray): Subsetted lon
        lat_subset (np.ndarray): Subsetted lat
    """
    lon_bounds = np.arange(-180, 180 + resolution_deg, resolution_deg)
    lat_bounds = np.arange(-90, 90 + resolution_deg, resolution_deg)

    # convert lon_range and lat_range to indices in the numpy arrays
    lon_start, lon_end = lon_range
    lat_start, lat_end = lat_range
    lon_idx = lambda lon: np.argmin(np.abs(lon_bounds - lon))
    lat_idx = lambda lat: np.argmin(np.abs(lat_bounds - lat))
    lon_start_idx, lon_end_idx = lon_idx(lon_start), lon_idx(lon_end)
    lat_start_idx, lat_end_idx = lat_idx(lat_start), lat_idx(lat_end)

    mask_subset = masks[:, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]
    var_subset = var[:, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]
    var_filtered_subset = var_filtered[
        :, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx
    ]
    lon_subset = lon_bounds[lon_start_idx : lon_end_idx + 1]
    lat_subset = lat_bounds[lat_start_idx : lat_end_idx + 1]
    if plot:
        fig, ax = plt.subplots()
        if mask_subset.ndim == 3:
            m = mask_subset[0]
            v = var_subset[0]
        elif mask_subset.ndim == 2:
            m = mask_subset
            v = var_subset
        ax.pcolormesh(lon_subset, lat_subset, m.T, vmin=0, vmax=2, cmap="viridis")
        ax.set_xlim(lon_start, lon_end)
        ax.set_ylim(lat_start, lat_end)
        ax.set_aspect(abs((lon_end - lon_start) / (lat_start - lat_end)) * 1.0)

    if save_folder is not None:
        all_years = sorted(set([d.year for d in dates]))
        year_str = (
            f"{all_years[0]}"
            if len(all_years) == 1
            else f"{min(all_years)}-{max(all_years)}"
        )
        lat_str = lat_range_to_str(lat_range)
        lon_str = lon_range_to_str(lon_range)
        save_path = os.path.join(
            save_folder,
            f"subset_pet_masks_with_{ssh_var}_{year_str}_lat{lat_str}_lon{lon_str}.npz",
        )
        np.savez_compressed(
            save_path,
            masks=mask_subset,
            dates=dates,
            var=var_subset,
            var_filtered=var_filtered_subset,
            lon_subset=lon_subset,
            lat_subset=lat_subset,
        )
        print(f"Saved mask subset to {save_path}")
    return mask_subset, var_subset, var_filtered_subset, lon_subset, lat_subset


def lon_range_to_str(lon_range):
    lon_start, lon_end = lon_range
    lon_start = f"{lon_start}E" if lon_start >= 0 else f"{abs(lon_start)}W"
    lon_end = f"{lon_end}E" if lon_end >= 0 else f"{abs(lon_end)}W"
    return f"{lon_start}-{lon_end}"


def lat_range_to_str(lat_range):
    lat_start, lat_end = lat_range
    lat_start = f"{lat_start}N" if lat_start >= 0 else f"{abs(lat_start)}S"
    lat_end = f"{lat_end}N" if lat_end >= 0 else f"{abs(lat_end)}S"
    return f"{lat_start}-{lat_end}"

def subset_arrays_for_test(
    masks,
    var,
    var_filtered,
    dates,
    lon_range,
    lat_range,
    resolution_deg,
    plot=False,
    ssh_var="adt",
    save_folder=None,
):
    """
    Subset the arrays to the given lon_range and lat_range.

    Args:
        masks (np.ndarray): Global eddy segmentation masks.
            Can be masks from multiple dates concatenated into one array
        var (np.ndarray): Global SSH value
        var_filtered (np.ndarray): Global SSH value after high-pass filter
        dates (list): List of `datetime` objects
        lon_range (tuple): Longitude range to subset to (lon_start, lon_end)
        lat_range (tuple): Latitude range to subset to (lat_start, lat_end)
        resolution_deg (float): Resolution of the SSH map in degrees
        plot (bool): Whether to plot a sample of the subsetted arrays
        ssh_var (str): SSH variable name. Defaults to "adt". Only used if save_folder is not None.
        save_folder (str): Folder to save the subsetted arrays to. Defaults to None.
            If None, the arrays are not saved.

    Returns:
        mask_subset (np.ndarray): Subsetted masks
        var_subset (np.ndarray): Subsetted var
        var_filtered_subset (np.ndarray): Subsetted var_filtered
        lon_subset (np.ndarray): Subsetted lon
        lat_subset (np.ndarray): Subsetted lat
    """
    lon_bounds = np.arange(-180, 180 + resolution_deg, resolution_deg)
    lat_bounds = np.arange(-90, 90 + resolution_deg, resolution_deg)

    # convert lon_range and lat_range to indices in the numpy arrays
    lon_start, lon_end = lon_range
    lat_start, lat_end = lat_range
    lon_idx = lambda lon: np.argmin(np.abs(lon_bounds - lon))
    lat_idx = lambda lat: np.argmin(np.abs(lat_bounds - lat))
    lon_start_idx, lon_end_idx = lon_idx(lon_start), lon_idx(lon_end)
    lat_start_idx, lat_end_idx = lat_idx(lat_start), lat_idx(lat_end)

    mask_subset = masks[:, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]
    var_subset = var[:, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]
    var_filtered_subset = var_filtered[
        :, lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx
    ]
    lon_subset = lon_bounds[lon_start_idx : lon_end_idx + 1]
    lat_subset = lat_bounds[lat_start_idx : lat_end_idx + 1]
    if plot:
        fig, ax = plt.subplots()
        if mask_subset.ndim == 3:
            m = mask_subset[0]
            v = var_subset[0]
        elif mask_subset.ndim == 2:
            m = mask_subset
            v = var_subset
        ax.pcolormesh(lon_subset, lat_subset, m.T, vmin=0, vmax=2, cmap="viridis")
        ax.set_xlim(lon_start, lon_end)
        ax.set_ylim(lat_start, lat_end)
        ax.set_aspect(abs((lon_end - lon_start) / (lat_start - lat_end)) * 1.0)

    if save_folder is not None:
        all_years = sorted(set([d.year for d in dates]))
        year_str = (
            f"{all_years[0]}"
            if len(all_years) == 1
            else f"{min(all_years)}-{max(all_years)}"
        )
        lat_str = lat_range_to_str(lat_range)
        lon_str = lon_range_to_str(lon_range)
        save_path = os.path.join(
            save_folder,
            f"subset_pet_masks_with_{ssh_var}_{year_str}_lat{lat_str}_lon{lon_str}.npz",
        )
        np.savez_compressed(
            save_path,
            masks=mask_subset,
            dates=dates,
            var=var_subset,
            var_filtered=var_filtered_subset,
            lon_subset=lon_subset,
            lat_subset=lat_subset,
        )
        print(f"Saved mask subset to {save_path}")
    return mask_subset, var_subset, var_filtered_subset, lon_subset, lat_subset