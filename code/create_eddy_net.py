import torch
from model_utils import EddyNet
from convert_to_pytorch_data_loader import *

num_classes = 2 if binary else 3
model_name = "eddynet"  # we'll log this in Tensorboard
model = EddyNet(num_classes, num_filters=16, kernel_size=3)
if torch.cuda.is_available():
    model.to(device="cuda")
