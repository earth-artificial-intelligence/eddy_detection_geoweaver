#Segmentation Model:

from eddy_import import *
from Eddy_Dataloader import *
import torch
#from models.eddynet import EddyNet
from eddynet import EddyNet
num_classes = 2 if binary else 3
model_name = "eddynet"  # we'll log this in Tensorboard
model = EddyNet(num_classes, num_filters=16, kernel_size=3)
if torch.cuda.is_available(): 
    model.to(device="cuda")
