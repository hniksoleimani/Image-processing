import cv2
import numpy as np

import matplotlib.pyplot as plt
img = cv2.imread('saf.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv,(10, 100, 20), (25, 25, 25))
BLUE_MIN = np.array([90, 100, 20],np.uint8)
BLUE_MAX = np.array([150, 255, 255],np.uint8)
frame_threshed = cv2.inRange(hsv, BLUE_MIN, BLUE_MAX)
# binarize the image
binr = cv2.threshold(frame_threshed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
# define the kernel
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN,kernel, iterations=1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
result = np.zeros(img.shape)
rows, cols, _ = img.shape
mask = np.zeros((rows,cols,1), dtype='uint8')

for i in range(rows):
    for j in range(cols):
        if closing[i,j] > 180:
            mask[i,j] = 1

out=img*mask


  
# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 2
columns = 2
  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)


# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Input Image")


  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(binr)
plt.axis('off')
plt.title("HSV Mask")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(mask)
plt.axis('off')
plt.title("Applying Morphology")
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(out)
plt.axis('off')
plt.title("Result")

plt.savefig('plot.png', dpi=300, bbox_inches='tight')






