import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')
#image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
            'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']

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


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    image = np.copy(img)
    bbox_list = []
    # Iterate through template list
    for template in template_list:
        # Read in templates one by one
        templ = mpimg.imread(template)
        # Use cv2.matchTemplate() to search the image
        # using whichever of the OpenCV search methods you prefer
        res = cv2.matchTemplate(image, templ, 3)
        # Use cv2.minMaxLoc() to extract the location of the best match
        # Determine bounding box corners for the match
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        # Return the list of bounding boxes
        maxLoc_bottom_right = (maxLoc[0] + templ.shape[1], maxLoc[1] + templ.shape[0])
        bbox_list.append((maxLoc, maxLoc_bottom_right))
    return bbox_list

bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
