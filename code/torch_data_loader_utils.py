# Write first python in Geoweaver
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import ArtistAnimation
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

torch.manual_seed(42)


def get_eddy_dataloader(
    files, binary=False, transform=None, batch_size=32, shuffle=True, val_split=0
):
    """
    Given a list of npz files, return dataloader(s) for train (and val).

    Args:
        files (list) : list of npz files
        binary (bool) : whether to use binary masks or not.
                        If True, treat cyclonic and anticyclonic eddies as single positive class.
        transform (callable) : optional transform to be applied on a sample.
        batch_size (int) : batch size for dataloader
        shuffle (bool) : whether to shuffle the dataset or not
        val_split (float) : fraction of data to be used as validation set.
                            If 0, no validation split is performed.
    Returns:
        (train_loader, val_loader) if val_split > 0; (train_loader, None) otherwise
    """
    ds, _ = get_eddy_dataset(files, binary, transform, val_split)
    loader_kwargs = dict(batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    if val_split > 0:
        train_ds, val_ds = ds
        train_dl = DataLoader(train_ds, **loader_kwargs)
        val_dl = DataLoader(val_ds, **loader_kwargs)
    else:
        train_dl = DataLoader(ds, **loader_kwargs)
        val_dl = None
    return train_dl, val_dl


def get_eddy_dataset(files, binary=None, transform=None, val_split=0):
    masks, dates, _, var_filtered, lon, lat, npz_dict = read_npz_files(files)
    print(f"Read {len(masks)} samples from {files}.")
    if val_split > 0:
        # split into training and validation sets (80% training, 20% validation)
        train_idx, val_idx = train_test_split(
            np.arange(len(masks)), test_size=val_split, random_state=42
        )
        train_ds = EddyDataset(
            masks[train_idx],
            var_filtered[train_idx],
            dates[train_idx],
            transform=transform,
            binary_mask=binary,
        )

        val_ds = EddyDataset(
            masks[val_idx],
            var_filtered[val_idx],
            dates[val_idx],
            transform=transform,
            binary_mask=binary,
        )
    else:
        train_ds = EddyDataset(
            masks, var_filtered, dates, transform=transform, binary_mask=binary
        )
        val_ds = None
    return train_ds, val_ds


def read_npz_files(npz_files: list):
    """Load a list of npz files, concatenate, and return separate arrays for eddy segmentation"""
    # load npz file into separate variables
    if isinstance(npz_files, str):
        npz_files = [npz_files]
    npz_contents = [np.load(file, allow_pickle=True) for file in npz_files]
    masks, dates, var, var_filtered, lon_subset, lat_subset = eddy_dict_to_vars(
        npz_contents
    )
    return masks, dates, var, var_filtered, lon_subset, lat_subset, npz_contents


def eddy_dict_to_vars(npz_contents):
    masks = np.concatenate(
        [npz_content["masks"] for npz_content in npz_contents], axis=0
    )
    dates = np.concatenate(
        [npz_content["dates"] for npz_content in npz_contents], axis=0
    )
    # var = np.concatenate([npz_content["var"] for npz_content in npz_contents], axis=0)
    var = None
    var_filtered = np.concatenate(
        [npz_content["var_filtered"] for npz_content in npz_contents], axis=0
    )
    if "lon_subset" in npz_contents[0]:
        lon_subset = np.concatenate(
            [npz_content["lon_subset"] for npz_content in npz_contents], axis=0
        )
        lat_subset = np.concatenate(
            [npz_content["lat_subset"] for npz_content in npz_contents], axis=0
        )
    else:
        lon_subset = lat_subset = None
    return masks, dates, var, var_filtered, lon_subset, lat_subset


class EddyDataset(torch.utils.data.Dataset):
    def __init__(self, masks, gv, dates, transform=None, binary_mask=False):
        """PyTorch dataset for eddy detection
        Args:
            masks (np.array): array of segmentation masks with shape: (N_dates, N_lon, N_lat)
                Can have 3 values: 0, 1 and 2, where 1 = anticyclonic, 2 = cyclonic and 0 = no eddy
            gv (np.array): array of GV maps with shape: (N_dates, N_lon, N_lat)
                Example GVs: sea level anomaly, absolute dynamic topography
            transform (callable, optional): Transformation to be applied on a sample.
            binary_mask (bool, optional): If true, all eddies (anticyclonic and cyclonic) will be assigned a value of 1
        """
        self.masks = masks
        self.gv = gv.astype(np.float32)  # GV stands for Geophysical Variable
        self.dates = dates
        self.transform = transform
        self.binary_mask = binary_mask

    def __getitem__(self, index, return_date=True):
        # return image and mask for a given index
        image = self.gv[index, :, :].copy()
        mask = self.masks[index, :, :].copy()
        date = self.dates[index]

        # transpose
        image = image.T
        mask = mask.T

        # address regions of land that are represented as -2147483648
        image[image < -10000] = 0

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # make ndim = 3

        if self.transform:
            image = self.transform(image)

        # if image and mask are numpy arrays, convert them to torch tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if self.binary_mask:
            mask[mask >= 1] = 1

        # convert to float
        image = image.float()

        if return_date:
            # convert date to tensor
            # date_str = date.strftime("%Y-%m-%d")
            # date =
            return image, mask, index
        else:
            return image, mask

    def __len__(self):
        return self.masks.shape[0]

    def plot_sample(self, N=5):

        # var in first column, mask in second column
        num_cols = 2
        num_rows = N
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
        ax[0, 0].set_title("GV")
        ax[0, 1].set_title("Mask")
        for i in range(num_rows):
            # get random sample from self
            n = np.random.randint(0, len(self))
            gv, mask, index = self.__getitem__(n, return_date=True)
            gv = np.squeeze(gv.cpu().detach().numpy())
            mask = np.squeeze(mask.cpu().detach().numpy())
            date = self.dates[index].strftime("%Y-%m-%d")
            # ax[i, 0].pcolormesh(lon_subset, lat_subset, gv.T, cmap="RdBu_r", vmin=-0.15, vmax=0.15)
            ax[i, 0].imshow(gv, cmap="RdBu_r", vmin=-0.15, vmax=0.15)
            ax[i, 0].set_title(f"GV ({date})")
            ax[i, 0].axis("off")
            ax[i, 1].imshow(mask, cmap="viridis")
            ax[i, 1].set_title(f"Mask ({date})")
            ax[i, 1].axis("off")

    def animate(self):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        print(f"Drawing animation of GV and segmentation mask")
        artists = []
        for i in tqdm(range(len(self)), desc="Animating eddies:"):
            gv, mask, date_idx = self.__getitem__(i, return_date=True)
            date = self.dates[date_idx].strftime("%Y-%m-%d")
            im1 = ax[0].imshow(gv.squeeze(), cmap="RdBu_r", vmin=-0.15, vmax=0.15)
            t1 = ax[0].text(
                0.5,
                1.05,
                f"GV {date}",
                size=plt.rcParams["axes.titlesize"],
                ha="center",
                transform=ax[0].transAxes,
            )
            ax[0].axis("off")

            im2 = ax[1].imshow(mask.squeeze(), cmap="viridis")
            t2 = ax[1].text(
                0.5,
                1.05,
                f"Mask {date}",
                size=plt.rcParams["axes.titlesize"],
                ha="center",
                transform=ax[1].transAxes,
            )
            ax[1].axis("off")
            plt.tight_layout()
            artists.append([im1, t1, im2, t2])
            fig.canvas.draw()
            fig.canvas.flush_events()
        animation = ArtistAnimation(fig, artists, interval=500, blit=True)
        plt.close()
        return animation

def transform_ssh(ssh_array):
    # normalize sea level anomaly between 0 and 1 based on min max
    ssh_array = (ssh_array - ssh_array.min()) / (ssh_array.max() - ssh_array.min())
    return ssh_array


# convert npy to compressed npz
def convert_npy_to_npz(npy_file):
    npz_file = npy_file.replace(".npy", ".npz")
    npy_contents = np.load(npy_file)
