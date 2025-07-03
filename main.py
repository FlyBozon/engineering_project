import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
import tifffile

from process_img import ImageProcessor 

image_path = './datasets/UAV_VisLoc_dataset/01/satellite01.tif'
processor = ImageProcessor(image_path)

processor.show()