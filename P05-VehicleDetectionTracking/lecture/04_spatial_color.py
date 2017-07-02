import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('cutout1.jpg')

# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH 
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space is 'HSV':
        features = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space is 'LUV':
        features = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    features = cv2.resize(img, size)
    features = features.ravel()
    # Return the feature vector
    return features
    
feature_vec = bin_spatial(image, color_space='HSV', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
