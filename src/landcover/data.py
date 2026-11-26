import numpy as np 
import rasterio 

# band order of the Sentinel files 
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B10", "B11", "B12"]

# function to the load the image 

def load_image(path):
    """Read a sentinel 2 image, returns a array of shape (bands, height, width)"""
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
    return image

def load_labels(path):
    """Reads a land cover tif. Returns an array of shape (heigh,width)"""
    with rasterio.open(path) as src: 
        labels = src.read(1)
        return labels 

def get_band(image, band_name):
    """Get one band by a name, e.g get_band(image, "B8"). Returns (height, width)"""
    return image[BANDS.index(band_name)]

def flatten_image(image):
    """(bands, height, width) -> (pixels, bands)"""
    bands, height, width = image.shape
    return image.reshape(bands,height * width).T

def flatten_labels(labels):
    """(height, width) -> (pixels,)"""
    return labels.flatten()

def unflatten_predictions(pred,height,width):
    """pixes -> (height, width), to plot the predicted map"""
    return pred.reshape(height,width)