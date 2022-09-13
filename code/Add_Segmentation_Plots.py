g, g_filtered, anticyclonic, cyclonic = identify_eddies(example_file, date)
ax, m = plot_variable(
    g_filtered, "adt", "Detected Eddies on ADT (m)", vmin=-0.15, vmax=0.15, cmap="Greys"
)
anticyclonic.display(
    ax, color="r", linewidth=0.75, label="Anticyclonic ({nb_obs} eddies)", ref=-180
)
cyclonic.display(
    ax, color="b", linewidth=0.75, label="Cyclonic ({nb_obs} eddies)", ref=-180
)
ax.legend()
update_axes(ax)

# Plot segmentation mask
mask = generate_segmentation_mask(
    g_filtered, anticyclonic, cyclonic, -180, 0, plot=True
)
