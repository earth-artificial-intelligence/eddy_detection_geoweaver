#Demonstrate the beginnings of how one can use classical computer vision techniques to recover eddy contours from the predicted segnmentation masks.

from eddy_import import *
from animation import *
p = preds[0].astype(np.uint8)

print(f"Number of anticyclonic eddies: {count_eddies(p, eddy_type='anticyclonic')}")
print(f"Number of cyclonic eddies: {count_eddies(p, eddy_type='cyclonic')}")
print(f"Number of both eddies: {count_eddies(p, eddy_type='both')}")

# draw contours on the image
thr = cv2.threshold(p, 0, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = np.zeros(p.shape, np.uint8)
cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
plt.imshow(img, cmap="gray")
plt.axis("off")

# get average contour area
area = 0
for cnt in contours:
    area += cv2.contourArea(cnt)
area /= len(contours)
print(f"Average contour area: {area:.2f} sq. pixels")
      
plt.savefig(f'{figOutputFolder}/EddyContours.png', bbox_inches ="tight")
