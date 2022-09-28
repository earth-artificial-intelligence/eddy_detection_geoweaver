# Defining the start_axes, update_axes, plot_variabe  and setting the paths for eddy workflow
from eddy_import import *

def start_axes(title):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
    ax.set_aspect("equal")
    ax.set_title(title, weight="bold")
    return ax


def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.94, 0.05, 0.01, 0.9]))


def plot_variable(grid_object, var_name, ax_title, **kwargs):
    ax = start_axes(ax_title)
    m = grid_object.display(ax, var_name, **kwargs)
    update_axes(ax, m)
    ax.set_xlim(grid_object.x_c.min(), grid_object.x_c.max())
    ax.set_ylim(grid_object.y_c.min(), grid_object.y_c.max())
    return ax, m

data_root = os.path.join(os.path.expanduser("~"), "ML_eddies")
train_folder = os.path.join(data_root, "cds_ssh_1998-2018_10day_interval")
test_folder = os.path.join(data_root, "cds_ssh_2019_10day_interval")

example_file = os.path.join(test_folder, "dt_global_twosat_phy_l4_20190101_vDT2021.nc")
date = datetime(2019, 1, 1)
g = RegularGridDataset(example_file, "longitude", "latitude")
