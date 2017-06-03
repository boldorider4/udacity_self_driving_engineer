import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
#%matplotlib inline
%matplotlib qt

image_path = 'output_images'

for file in [f for f in os.listdir(image_path) if \
             (f.startswith('straight_lines') or f.startswith('test'))]:

    image = mpimg.imread(os.path.join(image_path, file))
    #fig = plt.figure()
    #fig.suptitle('original')
    #plt.imshow(image)

    # Convert to HSV color space
    hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsl[:,:,0]
    l_channel = hsl[:,:,1]
    s_channel = hsl[:,:,2]

    # Convert to gray scale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.int16)

    # Thresholded sobel x on S channel
    sobelx_s = sobel_thresh(s_channel, 'x', 11, (40, 150))

    #fig = plt.figure()
    #fig.suptitle('sobelx_s')
    #plt.imshow(sobelx_s, cmap='gray')

    sobelx_gray = sobel_thresh(gray, 'x', 13, (30, 80))

    #fig = plt.figure()
    #fig.suptitle('sobelx_gray')
    #plt.imshow(sobelx_gray, cmap='gray')

    thresh_scale_s = scale_thresh_image(s_channel, (200, 255))

    #fig = plt.figure()
    #fig.suptitle('thresh_scale_s')
    #plt.imshow(thresh_scale_s, cmap='gray')

    thresh_scale_gray = scale_thresh_image(gray, (223, 255))

    #fig = plt.figure()
    #fig.suptitle('thresh_scale_gray')
    #plt.imshow(thresh_scale_gray, cmap='gray')

    dir_thresh_s = dir_thresh(s_channel, 7, (np.pi*40/180, np.pi*70/180))

    #fig = plt.figure()
    #fig.suptitle('dirpos_thresh_s')
    #plt.imshow(dir_thresh_s, cmap='gray')

    mag_thresh_gray = mag_thresh(gray, 13, (80, 180))

    #fig = plt.figure()
    #fig.suptitle('mag_thresh_gray')
    #plt.imshow(mag_thresh_gray, cmap='gray')

    binary_sum = np.zeros_like(gray)
    binary_sum[(thresh_scale_s == 1) | (thresh_scale_gray == 1) | (sobelx_s == 1)] = 1

    fig = plt.figure()
    fig.suptitle('binary_sum')
    plt.imshow(binary_sum, cmap='gray')

    binary_mask = np.zeros_like(gray)
    binary_mask[(sobelx_gray == 1) & (dir_thresh_s == 1)] = 1

    #fig = plt.figure()
    #fig.suptitle('binary_mask')
    #plt.imshow(binary_mask, cmap='gray')

    #overlay = np.zeros_like(image)
    #overlay[:,:,1] = thresh_scale_gray*255
    #image_overlay = np.copy(image)
    #image_overlay = cv2.addWeighted(image_overlay, .8, overlay, 0.7, 0.0)

    #fig = plt.figure()
    #fig.suptitle('image_overlay')
    #plt.imshow(image_overlay)
