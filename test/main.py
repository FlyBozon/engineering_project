import numpy as np
import matplotlib.pyplot as plt
#import rasterio
import cv2
#import tifffile

from process_img import ImageProcessor 
#from load_model import model

image_path = './datasets/UAV_VisLoc_dataset/01/satellite01.tif'
processor = ImageProcessor(image_path)

"""
processor.show()
tiles, poss = processor.split_into_tiles()
print("divided")
predictions = [processor.predict_tile(tile) for tile in tiles]
print("predicted")
final_res = processor.merge_predictions(predictions, poss, processor.image.shape)

unique_classes = np.unique(final_res)
print(f"Detected classes: {unique_classes}")
print(f"Class distribution:")
for cls in unique_classes:
    count = np.sum(final_res == cls)
    percentage = (count / final_res.size) * 100
    print(f"Class {cls}: {count} pixels ({percentage:.2f}%)")

processor.show(final_res, is_segmentation=True)
"""

tile, pos = processor.cut_one_tile(100, 12)
print("got one tile")
prediction = processor.predict_tile(tile)
print("predicted")
processor.show(prediction, is_segmentation=True, oryg_img=tile)