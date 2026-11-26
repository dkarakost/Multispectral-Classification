import numpy as np 
from landcover.data import get_band

def safe_devide(a,b):
    """a / b but returns 0 where b is 0 (avoids NaN and inf)"""
    out = np.zeros_like(a, dtype = np.float32)
    np.divide(a,b, out=out, where=(b != 0))
    return out 

def add_indices(image):
    """Take the (12, H, W) image, return (12 + n_indices, H, W)"""
    red = get_band(image,"B4")
    green = get_band(image, "B3")
    nir = get_band(image, "B8")
    swir1 = get_band(image, "B11")

    ndvi = safe_devide(nir -  red, nir + red) # ndvi index 
    ndwi = safe_devide(nir - green, nir + green) # ndwi index
    ndbi = safe_devide(swir1 - nir, swir1 + nir) # ndbi index

    new_features = np.stack([ndvi,ndwi,ndbi]) # stack those indices together with their values per pixel 
    image = np.concatenate([image, new_features], axis = 0)
    return image 

