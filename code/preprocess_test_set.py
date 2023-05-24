#Preprocessing for test files

from compression_and_segmentation_utils import *
from fetch_data_utils import *
from dependency import *


prev_date, prev_month, prev_year = get_dates_with_delta(331)

lon_range = (-166, -134)
lat_range = (14, 46)

data_root = os.path.join(os.path.expanduser("~"), root_path)

test_folder = os.path.join(data_root, "cds_ssh_test_everyday_interval")

file_pattern = "dt_global_twosat_phy_l4_{year:04d}{month:02d}{day:02d}_vDT2021.nc"

#

test_dates, test_files = get_dates_and_files(
    [int(prev_year)], [int(prev_month)], [int(prev_date)], test_folder, file_pattern
)


test_adt, test_adt_filtered, test_masks = generate_masks_in_parallel(
    test_files, test_dates, test=True
)

test_subset = subset_arrays_for_test(
    test_masks,
    test_adt,
    test_adt_filtered,
    test_dates,
    lon_range,
    lat_range,
    plot=True,
    resolution_deg=0.25,
    save_folder=test_folder,
    prev_date = int(prev_date),
    prev_month= int(prev_month),
    prev_year= int(prev_year)
)


