#Defining the subset arrays, converting the latitude and longitude range into indices in numpy, latitude range to str and longitude range to str.
from eddy_import import *
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
  
