from dependency import *

data_root = os.path.join(os.path.expanduser("~"), "ML_eddies")
train_folder = os.path.join(data_root, "cds_ssh_1998-2018_10day_interval")
test_folder = os.path.join(data_root, "cds_ssh_2019_10day_interval")

example_file = os.path.join(test_folder, "dt_global_twosat_phy_l4_20190110_vDT2021.nc")