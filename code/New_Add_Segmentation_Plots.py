#code for plotting segmentation masks, antcyclonic display, cyclonic display and updating the axis
from eddy_plots import *
from eddy_paths import *
from copy import deepcopy

#updated the r4ef details and also the vmin and vmax values
g, g_filtered, anticyclonic, cyclonic = identify_eddies(example_file, date)
ax, m = plot_variable(
    g_filtered, "adt", "Detected Eddies on ADT (m)", vmin=-5, vmax=5, cmap="Greys"
)
anticyclonic.display(
    ax, color="r", linewidth=0.75, label="Anticyclonic ({nb_obs} eddies)", ref=-250
)
cyclonic.display(
    ax, color="b", linewidth=0.75, label="Cyclonic ({nb_obs} eddies)", ref=-250
)
ax.legend()
update_axes(ax)

plt.savefig('/Users/lakshmichetana/ML_eddies_Output/Detected Eddies on ADT (m)_with_UpdatedVminVmax&RefValues.png', bbox_inches ="tight")

# Plot segmentation mask
mask = generate_segmentation_mask(
    g_filtered, anticyclonic, cyclonic, -180, 0, plot=True
)
plt.savefig(f'{figOutputFolder}/Segmentation Mask_with_UpdatedVminVmax&RefValues.png', bbox_inches ="tight")
