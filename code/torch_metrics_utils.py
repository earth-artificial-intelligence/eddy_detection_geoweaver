import torchmetrics
import torch
from create_eddy_net import *

def get_metrics(N, sync=False):
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
        torchmetrics.Accuracy(dist_sync_on_step=sync, num_classes=N,task="multiclass"),
        torchmetrics.Precision(
            average=None,
            dist_sync_on_step=sync,
            num_classes=N,
          	task="multiclass"
        ),
        torchmetrics.Recall(
            average=None,
            dist_sync_on_step=sync,
            num_classes=N,
          	task="multiclass"
        ),
#         torchmetrics.F1Score(  # TODO: Homework: verify in tensorboard that this is equivalent to accuracy
#             average="micro",
#             dist_sync_on_step=sync,
#             num_classes=N,
#         ),
        torchmetrics.F1Score(
            average="none",  # return F1 for each class
            dist_sync_on_step=sync,
            num_classes=N,
          	task="multiclass"
        )
    ]
    if torch.cuda.is_available():  # move metrics to the same device as model
        [metric.to("cuda") for metric in metrics]

    train_metrics = torchmetrics.MetricCollection(metrics)
    val_metrics = train_metrics.clone()
    return train_metrics, val_metrics

train_metrics, val_metrics = get_metrics(num_classes)

