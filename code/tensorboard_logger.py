#Tensor Logger
#We use the tensor logger to log our loss and metrics throughout the training process.

#from eddy_import import *
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

print('logger start')
tensorboard_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))),
    "tensorboard",
    # add current timestamp
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}",
)

print('logger middle')
writer = SummaryWriter(log_dir=tensorboard_dir)
print(
    f"{''.join(['=']*(28 + len(writer.log_dir)))}\n"
    f"Writing Tensorboard logs to {writer.log_dir}"
    f"\n{''.join(['=']*(28 + len(writer.log_dir)))}"
)

print('logger end')
