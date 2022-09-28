#Analyze training curves in TensorBoard

from eddy_import import *
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

%load_ext tensorboard
%tensorboard --bind_all --logdir $writer.log_dir --port=6008  # the default is 6006 but we set it to 6009 to avoid conflicts with other notebooks
