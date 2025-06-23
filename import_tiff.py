import rasterio
import matplotlib.pyplot as plt
src = rasterio.open('./datasets/UAV_VisLoc_dataset/01/satellite01.tif')
array = src.read(1)
#print(array.shape)
plt.imshow(array, cmap='ocean')
plt.show()