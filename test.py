'''
Make sure the active directory is the directory of the repo when running the test in a IDE
'''
from skimage.data import coins
import matplotlib.pyplot as plt
import MTM, cv2
import numpy as np

#%% Get image and templates by cropping
image     = coins()
smallCoin = image[37:37+38, 80:80+41]
bigCoin   = image[14:14+59,302:302+65]


#%% Perform matching
listHit  = MTM.findMatches([smallCoin, bigCoin], image)
tableHit = MTM.matchTemplates([('small', smallCoin), ('big', bigCoin)], image, score_threshold=0.6, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0) # Correlation-score
#tableHit = MTM.matchTemplates([('small', smallCoin), ('big', bigCoin)], image, score_threshold=0.4, method=cv2.TM_SQDIFF_NORMED, maxOverlap=0) # Difference-score

print("Found {} coins".format(len(tableHit)))
print(tableHit)

#%% Display matches
Overlay = MTM.drawBoxesOnRGB(image, tableHit, showLabel=True)
plt.figure()
plt.imshow(Overlay)

#%% Use GluonCV for display
import gluoncv as gcv

# Convert from x,y,w,h to xmin, ymin, xmax, ymax
BBoxes_xywh = np.array( tableHit["BBox"].tolist() ) 
BBoxes_xyxy = gcv.utils.bbox.bbox_xywh_to_xyxy(BBoxes_xywh)

Overlay2 = gcv.utils.viz.cv_plot_bbox(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), BBoxes_xyxy.astype("float64"), scores=tableHit["Score"].to_numpy(), thresh=0  )
plt.figure()
plt.imshow(Overlay2)