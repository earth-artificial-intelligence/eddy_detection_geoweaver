
import cv2  # use cv2 to count eddies by drawing contours around segmentation masks
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_device_config import *
from tqdm.auto import tqdm
from model_training_utils import run_batch, write_metrics_to_tensorboard, filter_scalar_metrics, EarlyStopping
from create_eddy_net import *

num_plots_in_tensorboard = 5
# will populate this later with random numbers:
random_plot_indices = np.zeros((num_plots_in_tensorboard,), np.uint8)


def run_epoch(
    epoch,
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    train_metrics,
    val_metrics,
    writer,
):
    leave = epoch == num_epochs - 1  # leave progress bar on screen after last epoch

    model.train()
    # training set
    for batch_num, (gvs, seg_masks, date_indices) in enumerate(train_loader):
        train_loss = run_batch(
            model, loss_fn, gvs, seg_masks, optimizer, scheduler, train_metrics
        )
        iter_num = epoch * len(train_loader) + batch_num
        writer.add_scalar("train/lr", scheduler.get_last_lr()[-1], iter_num)

    # validation set
    images, preds, labels, dates = [], [], [], []
    model.eval()
    with torch.no_grad():
        val_loss = num_examples = 0
        for gvs, masks, date_indices in val_loader:
            # continue
            loss_, pred_batch = run_batch(
                model, loss_fn, gvs, masks, metrics=val_metrics, return_pred=True
            )
            val_loss += loss_
            num_examples += np.prod(gvs.shape)
            # keep track of images, preds, labels for plotting
            images.append(gvs)
            preds.append(pred_batch)
            labels.append(masks)
            dates.append(date_indices)

    # calculate average validation loss across all samples
    # num_examples should be equal to sum of all pixels
    val_loss = val_loss / num_examples

    # plot validation images and log to tensorboard
    ## move images, preds, labels, dates to cpu
    images = torch.cat(images).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    dates = torch.cat(dates).cpu().numpy()
    ## convert indices to actual dates
    dates = [val_loader.dataset.dates[i].strftime("%Y-%m-%d") for i in dates]

    # take random images from validation set
    if epoch == 0:
        indices_ = np.random.choice(
            len(images), num_plots_in_tensorboard, replace=False
        )
        for i, idx in enumerate(indices_):
            random_plot_indices[i] = idx
    fig, ax = plt.subplots(num_plots_in_tensorboard, 3, figsize=(20, 30))
    for n, i in enumerate(random_plot_indices):
        date, img, mask, pred = dates[i], images[i], labels[i], preds[i]
        artists = plot_eddies_on_axes(
            date, img, mask, pred, ax[n, 0], ax[n, 1], ax[n, 2]
        )
    plt.tight_layout()
    writer.add_figure(f"val/sample_prediction", fig, global_step=epoch)

    # Update tensorboard
    train_m = write_metrics_to_tensorboard(
        num_classes, train_metrics, writer, epoch, "train"
    )
    val_m = write_metrics_to_tensorboard(num_classes, val_metrics, writer, epoch, "val")

    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)

    # reset metrics after each epoch
    train_metrics.reset()
    val_metrics.reset()

    train_m = filter_scalar_metrics(train_m)
    val_m = filter_scalar_metrics(val_m)

    return train_loss, val_loss, train_m, val_m


def plot_eddies_on_axes(date, img, mask, pred, a1, a2, a3):
    im1 = a1.imshow(img.squeeze(), cmap="viridis")

    # blit canvas for a1 a2 a3
    a1.figure.canvas.draw()
    a1.figure.canvas.flush_events()
    a2.figure.canvas.draw()
    a2.figure.canvas.flush_events()
    a3.figure.canvas.draw()
    a3.figure.canvas.flush_events()

    # https://stackoverflow.com/a/49159236
    t1 = a1.text(
        0.5,
        1.05,
        f"ADT {date}",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=a1.transAxes,
    )
    # set axis off
    a1.axis("off")

    # count number of eddies in mask and pred
    mask_anticyclonic = count_eddies(mask, "anticyclonic")
    mask_cyclonic = count_eddies(mask, "cyclonic")
    pred_anticyclonic = count_eddies(pred, "anticyclonic")
    pred_cyclonic = count_eddies(pred, "cyclonic")

    # calculate accuracy between pred and mask
    acc = np.sum(pred == mask) / mask.size
    im2 = a2.imshow(pred, cmap="viridis")
    t2 = a2.text(
        0.5,
        1.05,
        (
            f"Prediction (Acc = {acc:.3f} |"
            f" Num. anticyclonic = {pred_anticyclonic} |"
            f" Num. cyclonic = {pred_cyclonic})"
        ),
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=a2.transAxes,
    )
    a2.axis("off")
    im3 = a3.imshow(mask, cmap="viridis")
    t3 = a3.text(
        0.5,
        1.05,
        (
            f"Ground Truth"
            f" (Num. anticyclonic: {mask_anticyclonic} |"
            f" Num. cyclonic: {mask_cyclonic})"
        ),
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=a3.transAxes,
    )
    a3.axis("off")

    return im1, t1, im2, t2, im3, t3


def count_eddies(arr, eddy_type="both"):
    mask = np.zeros(arr.shape, dtype=np.uint8)
    if eddy_type == "anticyclonic":
        mask[arr == 1] = 1
    elif eddy_type == "cyclonic":
        mask[arr == 2] = 1
    else:
        mask[arr > 0] = 1
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)
