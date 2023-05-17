#predict eddies on testdataset

from matplotlib.animation import ArtistAnimation
from model_components import *
from training_and_plot_utils  import *
from device_config_and_data_loader import *
import os
from fetch_data_utils import *

prev_date, prev_month, prev_year = get_dates_with_delta(331)

data_root = os.path.join(os.path.expanduser("~"), "ML_test")

val_folder = os.path.join(data_root, "cds_ssh_test_everyday_interval")

prev_date = int(prev_date)
prev_month = int(prev_month)
prev_year = int(prev_year)

val_file = os.path.join(val_folder, f"subset_pet_masks_with_adt_{prev_year:04d}{prev_month:02d}{prev_date:02d}_lat14N-46N_lon166W-134W.npz")

binary = False
num_classes = 2 if binary else 3

val_loader, _ = get_eddy_dataloader(
    val_file, binary=binary, batch_size=batch_size, shuffle=False
)

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

print(os.path.join(tensorboard_dir, "test_predictions.gif"))
animation.save(os.path.join(tensorboard_dir, "test_predictions.gif"), writer="pillow")

# HTML(animation.to_jshtml())

#plot contour

p = preds[0].astype(np.uint8)

print(f"Number of anticyclonic eddies: {count_eddies(p, eddy_type='anticyclonic')}")
print(f"Number of cyclonic eddies: {count_eddies(p, eddy_type='cyclonic')}")
print(f"Number of both eddies: {count_eddies(p, eddy_type='both')}")

# draw contours on the image
thr = cv2.threshold(p, 0, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = np.zeros(p.shape, np.uint8)
cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
fileName = os.path.join("/home/chetana/plots/test/",f"{prev_year:04d}{prev_month:02d}{prev_date:02d}_contours.png")
cv2.imwrite(fileName, img)
plt.imshow(img, cmap="gray")
plt.axis("off")

# get average contour area
area = 0
      
      
for cnt in contours:
    area += cv2.contourArea(cnt)
area /= len(contours)
      
      
print(f"Average contour area: {area:.2f} sq. pixels")


