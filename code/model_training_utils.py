import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.tensorboard.summary import hparams


def run_batch(
    model,
    loss_fn,
    x_batch,
    y_batch,
    opt=None,
    sched=None,
    metrics=None,
    return_pred=False,
):
    """Run a batch of data through the model and return loss and metrics."""
    if torch.cuda.is_available():
        loss_fn = loss_fn.to(device="cuda")
        x_batch = x_batch.to(device="cuda", non_blocking=True)
        y_batch = y_batch.to(device="cuda", non_blocking=True)

    # forward pass
    logits = model(x_batch)
    if return_pred:
        preds = logits.argmax(axis=1).squeeze()
    # reshape so that each pixel in seg. mask can be treated as separate instance
    mask_flattened, logits = reshape_mask_and_predictions(y_batch, logits)
    # compute loss
    loss = loss_fn(logits, mask_flattened)
    # backward pass
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        if sched is not None:
            sched.step()
    # update metrics
    if metrics is not None:
        metrics.update(logits, mask_flattened)
    if return_pred:
        return loss.item(), preds
    else:
        return loss.item()


def reshape_mask_and_predictions(mask, prediction):
    """flatten mask and prediction in each batch"""
    mask_reshaped = mask.flatten().to(torch.int64)
    # pred_reshaped = prediction.flatten(start_dim=-2, end_dim=-1)
    # logits shape: [B, C, 128, 128] -> [B, 128, 128, C] -> [B * 128 * 128, C]
    pred_reshaped = prediction.permute((0, 2, 3, 1)).flatten(start_dim=0, end_dim=-2)
    return mask_reshaped, pred_reshaped


def get_metrics(N, sync):
    """Get the metrics to be used in the training loop.
    Args:
        N (int): The number of classes.
        sync (bool): Whether to use wait for metrics to sync across devices before computing value.
    Returns:
        train_metrics (MetricCollection): The metrics to be used in the training loop.
        val_metrics (MetricCollection): The metrics to be used in validation.
    """
    # Define metrics and move to GPU if available
    metrics = [
        torchmetrics.Accuracy(dist_sync_on_step=sync, num_classes=N),
        torchmetrics.Precision(
            average=None,
            dist_sync_on_step=sync,
            num_classes=N,
        ),
        torchmetrics.Recall(
            average=None,
            dist_sync_on_step=sync,
            num_classes=N,
        ),
        torchmetrics.F1Score(
            average="micro",
            dist_sync_on_step=sync,
            num_classes=N,
        ),
        # torchmetrics.AUROC(dist_sync_on_step=sync, num_classes=N),
        # StorePredLabel(dist_sync_on_step=sync),
    ]
    if torch.cuda.is_available():  # move metrics to the same device as model
        [metric.to("cuda") for metric in metrics]

    train_metrics = torchmetrics.MetricCollection(metrics)
    val_metrics = train_metrics.clone()
    return train_metrics, val_metrics


def write_metrics_to_tensorboard(N, metrics, writer, epoch, train_or_val):
    m = metrics.compute()
    for k, v in m.items():
        if k == "StorePredLabel":
            pred, label = v
            label = nn.functional.one_hot(label, N)
            writer.add_pr_curve(f"{train_or_val}/pr_curve", label, pred, epoch)
        # handle class-level metrics
        elif isinstance(v, torch.Tensor) and len(v.shape) > 0 and v.shape[-1] > 1:
            for i, v_ in enumerate(v):
                if N == 2:  # binary
                    l = "negative" if i == 0 else "positive"
                elif N == 3:
                    if i == 0:
                        l = "negative"
                    elif i == 1:
                        l = "anticyclonic"
                    elif i == 2:
                        l = "cyclonic"
                else:
                    raise NotImplementedError(f"{N} classes not supported")
                writer.add_scalar(f"{train_or_val}/{k}_{l}", v_, epoch)
        else:
            writer.add_scalar(f"{train_or_val}/{k}", v, epoch)
    return m


def filter_scalar_metrics(metrics_dict):
    """Filters the output of metrics.compute() and returns only the scalar metrics."""
    output = {}
    for k, v in metrics_dict.items():
        if (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)) and len(
            v.shape
        ) == 0:
            output[k] = v
    return output


def add_hparams(
    torch_tb_writer, hparam_dict, metric_dict, hparam_domain_discrete=None, epoch_num=0
):
    """Add a set of hyperparameters to be compared in TensorBoard.
    Args:
        hparam_dict (dict): Each key-value pair in the dictionary is the
            name of the hyper parameter and it's corresponding value.
            The type of the value can be one of `bool`, `string`, `float`,
            `int`, or `None`.
        metric_dict (dict): Each key-value pair in the dictionary is the
            name of the metric and it's corresponding value. Note that the key used
            here should be unique in the tensorboard record. Otherwise the value
            you added by ``add_scalar`` will be displayed in hparam plugin. In most
            cases, this is unwanted.
        hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
            contains names of the hyperparameters and all discrete values they can hold
    """
    torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError("hparam_dict and metric_dict should be dictionary.")
    exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

    torch_tb_writer.file_writer.add_summary(exp)
    torch_tb_writer.file_writer.add_summary(ssi)
    torch_tb_writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        torch_tb_writer.add_scalar(k, v, epoch_num)


# Taken from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        min_epochs=30,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.min_epochs = min_epochs
        self.epochs = 0

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience and self.epochs >= self.min_epochs:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        self.epochs += 1

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
