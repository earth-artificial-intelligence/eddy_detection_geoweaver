#Preprocessing for test files

from segmask_and_ssh_utils import *
from generate_ground_truth_parallel_utils import *

print("process is here")

lon_range = (-166, -134)
lat_range = (14, 46)

data_root = os.path.join(os.path.expanduser("~"), "ML_eddies")

test_folder = os.path.join(data_root, "cds_ssh_2019_10day_interval")

file_pattern = "dt_global_twosat_phy_l4_{year:04d}{month:02d}{day:02d}_vDT2021.nc"

print("testdate before:", test_folder)

test_dates, test_files = get_dates_and_files(
    [2019], [1], [10], test_folder, file_pattern
)


test_adt, test_adt_filtered, test_masks = generate_masks_in_parallel(
    test_files, test_dates, test=True
)

test_subset = subset_arrays(
    test_masks,
    test_adt,
    test_adt_filtered,
    test_dates,
    lon_range,
    lat_range,
    plot=True,
    resolution_deg=0.25,
    save_folder=test_folder,
)

