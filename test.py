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

listLabels = ["small", "big"]
listTemplates = [smallCoin, bigCoin]


#%% Perform matching
listHit      = MTM.findMatches(image, listTemplates, listLabels)
singleObject = MTM.findMatches(image, listTemplates, listLabels, nObjects=1)  # there should be 1 top hit per template

finalHits = MTM.matchTemplates(image,
                               listTemplates,
                               listLabels,
                               score_threshold=0.4,
                               maxOverlap=0)

print("Found {} coins".format(len(finalHits)))
print (np.array(finalHits)) # better formatting with array

#%% Display matches
MTM.plotDetections(image, finalHits, showLegend=True)