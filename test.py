'''
Make sure the active directory is the directory of the repo when running the test in a IDE
'''
from skimage.data import coins
import matplotlib.pyplot as plt
import MTM
print( MTM.__version__ )
import numpy as np

#%% Get image and templates by cropping
image     = coins()
smallCoin = image[37:37+38, 80:80+41]
bigCoin   = image[14:14+59,302:302+65]

asHigh  = image[:,10:50]
asLarge = image[50:70,:]

listTemplate = [smallCoin, bigCoin]


#%% Perform matching
listHit      = MTM.findMatches(listTemplate, image)
singleObject = MTM.findMatches(listTemplate, image, nObjects=1) # there should be 1 top hit per template

finalHits    = MTM.matchTemplates(listTemplate, image, score_threshold=0.6, maxOverlap=0) 

print("Found {} coins".format(len(finalHits)))
print(finalHits)

#%% Display matches
overlay = MTM.drawBoxesOnRGB(image, finalHits, thickness=1)
plt.figure()
plt.imshow(overlay)

#%% Use GluonCV for display
import gluoncv as gcv

"""
# for loop needed
# Convert from x,y,w,h to xmin, ymin, xmax, ymax
BBoxes_xywh = np.array( finalHits["BBox"].tolist() ) 
BBoxes_xyxy = gcv.utils.bbox.bbox_xywh_to_xyxy(BBoxes_xywh)

Overlay2 = gcv.utils.viz.cv_plot_bbox(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), BBoxes_xyxy.astype("float64"), scores=finalHits["Score"].to_numpy(), thresh=0  )
plt.figure()
plt.imshow(Overlay2)
"""