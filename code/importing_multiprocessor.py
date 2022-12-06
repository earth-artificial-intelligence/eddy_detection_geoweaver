#Code for generating the masks in parallel and getting dates and files

from eddy_import import *
from eddy_plots import *
import multiprocessing

def generate_masks_in_parallel(
    files,
    dates,
    ssh_var="adt",
    u_var="ugosa",
    v_var="vgosa",
    high_pass_wavelength_km=700,
    x_offset=-180,
    y_offset=0,
    num_processes=1,
    plot=False,
    save=True,
):
    pool = multiprocessing.Pool(processes=20)
    args = [
        (file, date, ssh_var, u_var, v_var, high_pass_wavelength_km, x_offset, y_offset)
        for file, date in zip(files, dates)
    ]
    #pool = multiprocessing.Pool(processes=num_processes)
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

    pool.close()
    pool.join()
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
