#import rasterio
import matplotlib.pyplot as plt

import cv2
import tifffile
import matplotlib.pyplot as plt

image_path = './datasets/UAV_VisLoc_dataset/01/satellite01.tif'

def dec_title(image, title_size = 512, overlay = 64):
    h, w = image.shape[:2]
    title = []
    poss = []
    for y in range(0, h-title_size + 1, title_size - overlay):
        for x in range (0, w-title_size+1, title_size-overlay):
            tmp_title = image[y:y+title_size, x:x+title_size]
            title.append(tmp_title)
            poss.append((x,y))
    return title, poss

src = rasterio.open(image_path)
array = src.read(1)
#print(array.shape)
# plt.imshow(array)
# plt.show()

img_tiff = tifffile.imread(image_path)
img_cv2=cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)) #imread read in BGR instead of RGB, so image needs to be converted to RGB before showing
# plt.show()

# plt.imshow(img_tiff)
# plt.show()

title, pos = dec_title(img_tiff)
plt.imshow(title[1])
plt.show()
plt.imshow(title[2])
plt.show()
plt.imshow(title[3])
plt.show()
plt.imshow(title[4])
plt.show()
plt.imshow(title[5])
plt.show()
plt.imshow(title[6])
plt.show()