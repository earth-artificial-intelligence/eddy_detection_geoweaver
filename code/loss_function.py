#loss function
import torch
from eddy_import import *
loss_fn = torch.nn.CrossEntropyLoss()
# TODO (homework): Try 
# loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(total_pixels/class_frequency))
