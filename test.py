"""
TEST SCRIPT.

Make sure the active directory is the directory of the repo when running the test in a IDE
"""
#%%
from skimage.data import coins
import matplotlib.pyplot as plt
import MTM, cv2
import numpy as np

print(MTM.__version__)

#%% Get image and templates by cropping
image     = coins()

smallCoin = image[37:37+38, 80:80+41]
bigCoin   = image[14:14+59,302:302+65]
listTemplates = [('small', smallCoin), 
                 ('big', bigCoin)]


#%% Perform matching
listHit = MTM.matchTemplates(listTemplates, image, score_threshold=0.3, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0)  # Correlation-score
#tableHit = MTM.matchTemplates(listTemplates, image, score_threshold=0.4, method=cv2.TM_SQDIFF_NORMED, maxOverlap=0) # Difference-score

print("Found {} coins".format(len(listHit)))
print(np.array(listHit, dtype=object))


#%% Display matches
Overlay = MTM.drawBoxesOnRGB(image, listHit, showLabel=True)
plt.figure()
plt.imshow(Overlay)

#%% Check that it is raising an error if template is larger than image or search region
# Comment all except one to test them
MTM.matchTemplates(listTemplates, image, searchBox=(0,0,20,20)) # larger than search region
MTM.matchTemplates(listTemplates, 
                   image, 
                   searchBox=(0,0) + bigCoin.shape[::-1]) # as large as search region

tooLarge = ("tooLarge", np.pad(image, 1))
MTM.matchTemplates([tooLarge], image) # larger than image

#%% Use GluonCV for display
import gluoncv as gcv

# "Unzip" the list of tuple, into individual lists
listLabel, listBbox, listScore = zip(*listHit)

# Convert from x,y,w,h to xmin, ymin, xmax, ymax
BBoxes_xywh = np.array(listBbox) 
BBoxes_xyxy = gcv.utils.bbox.bbox_xywh_to_xyxy(BBoxes_xywh)

Overlay2 = gcv.utils.viz.cv_plot_bbox(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), BBoxes_xyxy.astype("float64"), scores=listScore, thresh=0  )
plt.figure()
plt.imshow(Overlay2)