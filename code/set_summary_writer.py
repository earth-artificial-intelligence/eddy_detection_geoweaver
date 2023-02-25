
import datetime
from torch.utils.tensorboard import SummaryWriter

tensorboard_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))),
    "tensorboard",
    # add current timestamp
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}",
)
writer = SummaryWriter(log_dir=tensorboard_dir)
print(
    f"{''.join(['=']*(28 + len(writer.log_dir)))}\n"
    f"Writing Tensorboard logs to {writer.log_dir}"
    f"\n{''.join(['=']*(28 + len(writer.log_dir)))}"
)

