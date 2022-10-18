#Using plot_sample to visualize the dataset we just loaded.
from eddy_import import *
from get_eddy_dataloader import *
train_loader.dataset.plot_sample(N=3)

plt.savefig("/Users/lakshmichetana/ML_eddies_Output/datasetPlots",bbox="tight")
