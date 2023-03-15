from matplotlib.animation import ArtistAnimation
from model_utils import *
from set_summary_writer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/home/chetana/tensorboard/2023-03-15_03-26/model_ckpt_68.pt"))
model.eval()
with torch.no_grad():
    fig, ax = plt.subplots(1, 3, figsize=(25, 10))
    artists = []
    # loop through all SSH maps and eddy masks in 2019
    # and run the model to generate predicted eddy masks
    for n, (ssh_vars, seg_masks, date_indices) in enumerate(val_loader):
        ssh_vars = ssh_vars.to(device)
        seg_masks = seg_masks.to(device)
        # Run the model to generate predictions
        preds = model(ssh_vars)

        # For each pixel, EddyNet outputs predictions in probabilities,
        # so choose the channels (0, 1, or 2) with the highest prob.
        preds = preds.argmax(dim=1)

        # Loop through all SSH maps, eddy masks, and predicted masks
        # in this minibatch and generate a video
        preds = preds.cpu().numpy()
        seg_masks = seg_masks.cpu().numpy()
        ssh_vars = ssh_vars.cpu().numpy()
        date_indices = date_indices.cpu().numpy()
        for i in range(len(ssh_vars)):
            date, img, mask, pred = date_indices[i], ssh_vars[i], seg_masks[i], preds[i]
            img1, title1, img2, title2, img3, title3 = plot_eddies_on_axes(
                date, img, mask, pred, ax[0], ax[1], ax[2]
            )
            artists.append([img1, title1, img2, title2, img3, title3])
            fig.canvas.draw()
            fig.canvas.flush_events()
    animation = ArtistAnimation(fig, artists, interval=200, blit=True)
    plt.close()

print(os.path.join(tensorboard_dir, "val_predictions.gif"))
animation.save(os.path.join(tensorboard_dir, "val_predictions.gif"), writer="pillow")
# HTML(animation.to_jshtml())
