# This file contanis differetnt components to run the model
# 1. Eddy-Net Object
# 2. Torch Metrics (metrics to keep track of)
# 3. Optimizer
# 4. Loss function
# 5. Summery writer to write summary

import torchmetrics
import torch
import datetime
import os

from torch.utils.tensorboard import SummaryWriter
from data_utils import EddyNet
from convert_to_pytorch_data_loader import *

# 1. Eddy Net
num_classes = 2 if binary else 3
model_name = "eddynet"  # we'll log this in Tensorboard
model = EddyNet(num_classes, num_filters=16, kernel_size=3)
if torch.cuda.is_available():
    model.to(device="cuda")


# 2. Torch Metrics (metrics to keep track of)
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



# 3. Optimizer
initial_lr = 1e-6
max_lr = 5e-4

optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    div_factor=max_lr / initial_lr,
    pct_start=0.3,
)


# 4. Loss function
loss_fn = torch.nn.CrossEntropyLoss()


# 5. Summery writer
tensorboard_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))),
    "tensorboard",
    # add current timestamp
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}",
)
writer = SummaryWriter(log_dir=tensorboard_dir)
print(
    f"{''.join(['=']*(28 + len(writer.log_dir)))}\\n"
    f"Writing Tensorboard logs to {writer.log_dir}"
    f"\\n{''.join(['=']*(28 + len(writer.log_dir)))}"
)




