import torch
from convert_to_pytorch_data_loader import *
from create_eddy_net import *

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

