'''
Make sure the active directory is the directory of the repo when running the test in a IDE
'''
from skimage.data import chelsea
import matplotlib.pyplot as plt
import MTM
print( MTM.__version__ )
import numpy as np

#%% Get image and templates by cropping
image     = chelsea()
template = image[80:150, 130:210]

listTemplates = [template]


#%% Perform matching
#listHit = MTM.findMatches(image, listTemplates)
#listHit = MTM.findMatches(image, listTemplates, nObjects=1)  # there should be 1 top hit per template

finalHits = MTM.matchTemplates(image,
                               listTemplates,
                               score_threshold=0.6,
                               maxOverlap=0,
                               nObjects=10)

print("Found {} detections".format(len(finalHits)))
print (np.array(finalHits)) # better formatting with array

#%% Display matches
MTM.plotDetections(image, finalHits)