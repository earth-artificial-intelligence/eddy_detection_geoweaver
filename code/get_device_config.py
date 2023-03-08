from dependency import *
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # useful on multi-GPU systems with multiple users

# Fix manual seeds for reproducibility
import torch
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

num_epochs = 250  # can lower this to save time
batch_size = 256
