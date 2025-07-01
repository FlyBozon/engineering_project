import rasterio
import matplotlib.pyplot as plt

import cv2
import tifffile
import matplotlib.pyplot as plt

image_path = './datasets/UAV_VisLoc_dataset/01/satellite01.tif'

src = rasterio.open(image_path)
array = src.read(1)
#print(array.shape)
plt.imshow(array)
plt.show()

img_tiff = tifffile.imread(image_path)
img_cv2=cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)) #imread read in BGR instead of RGB, so image needs to be converted to RGB before showing
plt.show()

plt.imshow(img_tiff)
plt.show()