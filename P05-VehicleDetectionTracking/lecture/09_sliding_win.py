import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    x_span = x_start_stop[1]-x_start_stop[0]
    y_span = y_start_stop[1]-y_start_stop[0]
    # Compute the number of pixels per step in x/y
    x_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    y_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    n_win_x = np.int((x_span-xy_window[0]*xy_overlap[0])/x_step)
    n_win_y = np.int((y_span-xy_window[1]*xy_overlap[1])/y_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    y_scan = y_start_stop[0]
    for windowy in range(n_win_y):
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        x_scan = x_start_stop[0]
        # Calculate each window position
        # Append window position to list
        for windowx in range(n_win_x):
            bbox = ((x_scan, y_scan), \
                (x_scan+xy_window[0], y_scan+xy_window[1]))
            window_list.append(bbox)
            x_scan += x_step
        y_scan += y_step
    # Return the list of windows
    return window_list

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                       
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
plt.imshow(window_img)
