import numpy as np
import matplotlib.pyplot as plt
#import rasterio
import cv2
#import tifffile

from process_img import ImageProcessor 
#from load_model import model

image_path = './datasets/UAV_VisLoc_dataset/01/satellite01.tif'
processor = ImageProcessor(image_path)

processor.show()
tiles, poss = processor.split_into_tiles()
print("divided")
predictions = [processor.predict_tile(tile) for tile in tiles]
print("predicted")
final_res = processor.merge_predictions(predictions, poss, processor.image.shape)
processor.show(final_res)