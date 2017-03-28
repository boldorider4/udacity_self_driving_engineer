import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from mpl_toolkits.axes_grid1 import ImageGrid
from helper_functions import *

fig_idx = 0

# Region of interest mask
imshape = np.array([540, 960], dtype=np.int32)
vertices = np.array([[(50,imshape[0]),(460, 320), (500, 320), (910,imshape[0])]], dtype=np.int32)

# Gaussian and Canny filters
gaussian_size = 3
canny_min = 50
canny_max = 150

# Hough transform
rho = 1
theta = np.pi/180
threshold = 25
min_line_len = 30; max_line_gap = 20

for file in [f for f in os.listdir('../test_images') if f.endswith('.jpg')]:

    image_filename = os.path.join('../test_images', file)
    print('Showing pipeline for image ' + image_filename)
    image = mpimg.imread(image_filename)
    gray_image = grayscale(image)
    gausblur_image = gaussian_blur(gray_image, gaussian_size)
    canny_image = canny(gausblur_image, canny_min, canny_max)

    roi_image = region_of_interest(canny_image, vertices)
    image_lines, lines = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)

    color_canny = np.dstack((canny_image, canny_image, canny_image))
    lines_over_canny = weighted_img(image_lines, color_canny, 0.8, .5, 0.)
    lines_over_image = weighted_img(image_lines, image, 0.8, .5, 0.)

    fig_idx += 1;
    fig = plt.figure(fig_idx, (15., 10.))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=.5)
    grid[0].imshow(image)
    grid[1].imshow(roi_image, cmap='gray')
    grid[2].imshow(lines_over_canny)
    grid[3].imshow(lines_over_image)
    plt.show()
