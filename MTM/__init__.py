import cv2
import numpy  as np
from skimage.feature import peak_local_max, match_template
from scipy.signal    import find_peaks

from .NMS import NMS

__all__ = ['NMS']
__version__ = '1.5.4'

def _findLocalMax_(corrMap, score_threshold=0.6):
    """Get coordinates of the local maximas with values above a threshold in the image of the correlation map."""
    # IF depending on the shape of the correlation map
    if corrMap.shape == (1,1): ## Template size = Image size -> Correlation map is a single digit')
        
        if corrMap[0,0]>=score_threshold:
            listPeaks = np.array([[0,0]])
        else:
            listPeaks = []
    
    else: # Correlation map is 2D
        listPeaks = peak_local_max(corrMap, threshold_abs=score_threshold, exclude_border=False).tolist()
        
    return listPeaks



def _findLocalMin_(corrMap, score_threshold=0.4):
    """Find coordinates of local minimas with values below a threshold in the image of the correlation map."""
    return _findLocalMax_(-corrMap, -score_threshold)


def computeScoreMap(template, image, method=cv2.TM_CCOEFF_NORMED):
    """
    Compute score map provided numpy array for template and image.
    
    Automatically converts images if necessary
    return score map as numpy as array
    """
    if template.dtype == "float64" or image.dtype == "float64": 
        raise ValueError("64-bit not supported, max 32-bit")
        
    # Convert images if not both 8-bit (OpenCV matchTempalte is only defined for 8-bit OR 32-bit)
    if not (template.dtype == "uint8" and image.dtype == "uint8"):
        template = np.float32(template)
        image    = np.float32(image)
    
    # Compute correlation map
    return cv2.matchTemplate(template, image, method)


def findMatches(listTemplates, image, nObjects=float("inf"), score_threshold=0.5, searchBox=None):
    """
    Find all possible templates locations provided a list of template to search and an image.
    
    Parameters
    ----------
    - listTemplates : list of templates as grayscale or RGB numpy array
                      templates to search in each image 
    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates

    - nObjects: int
                expected number of objects in the image
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    - searchBox : tuple (X, Y, Width, Height) in pixel unit
                optional rectangular search region as a tuple
    
    Returns
    -------
    - list of hits encoded as [template index, score, (x,y,width, height)]
    """
    if nObjects!=float("inf") and type(nObjects)!=int:
        raise TypeError("nObjects must be an integer")
        
    elif nObjects<1:
        raise ValueError("At least one object should be expected in the image")
        
    ## Crop image to search region if provided
    if searchBox != None: 
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset:yOffset+searchHeight, xOffset:xOffset+searchWidth]
    else:
        xOffset=yOffset=0
    
    listHit = []
    for index, template in enumerate(listTemplates):
        
        #print('\nSearch with template : ',templateName)
        
        #corrMap = computeScoreMap(template, image, method)
        corrMap = match_template(image, template)
        
        ## Find possible location of the object 
        if nObjects==1: # Detect global Min/Max
            #minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(corrMap)
            #listPeaks = [maxLoc[::-1]]
            listPeaks = peak_local_max(corrMap, threshold_abs=score_threshold, exclude_border=False, num_peaks=1).tolist()
            #listPeaks = np.unravel_index(np.argmax(corrMap), corrMap.shape)
            
        else:# Detect local max or min
            listPeaks = _findLocalMax_(corrMap, score_threshold)
            
        #print('Initially found',len(listPeaks),'hit with this template')
        
        # Once every peak was detected for this given template
        ## Create a dictionnary for each hit with {'TemplateName':, 'BBox': (x,y,Width, Height), 'Score':coeff}
        
        height, width = template.shape[0:2] # slicing make sure it works for RGB too
        
        for peak in listPeaks :
            score = corrMap[tuple(peak)]
            bbox  = int(peak[1]) + xOffset, int(peak[0]) + yOffset, width, height
            hit   = [score, bbox, index]

            # append to list of potential hit before Non maxima suppression
            listHit.append(hit)
    
    return listHit # All possible hits before Non-Maxima Supression
    

def matchTemplates(listTemplates, image, nObjects=float("inf"), score_threshold=0.5, maxOverlap=0.25, searchBox=None):
    """
    Search each template in the image, and return the best nObjects location which offer the best score and which do not overlap.
    
    Parameters
    ----------
    - listTemplates : list of templates as 2D grayscale or RGB numpy array
                      templates to search in each image, associated to a label 
    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates
    - nObjects: int
                expected number of objects in the image
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    - maxOverlap: float in range [0,1]
                This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
                If the ratio is over the maxOverlap, the lower score bounding box is discarded.
    - searchBox : tuple (X, Y, Width, Height) in pixel unit
                optional rectangular search region as a tuple
    
    Returns
    -------
    Pandas DataFrame with 1 row per hit and column "TemplateName"(string), "BBox":(X, Y, Width, Height), "Score":float                 
        if N=1, return the best matches independently of the score_threshold
        if N<inf, returns up to N best matches that passed the score_threshold
        if N=inf, returns all matches that passed the score_threshold
    """
    if maxOverlap<0 or maxOverlap>1:
        raise ValueError("Maximal overlap between bounding box is in range [0-1]")
        
    listHit  = findMatches(listTemplates, image, nObjects, score_threshold, searchBox)
    bestHits = NMS(listHit, maxOverlap, nObjects)
    
    return bestHits


def drawBoxesOnRGB(image, tableHit, boxThickness=2, boxColor=(255, 255, 00), showLabel=False, labelColor=(255, 255, 0), labelScale=0.5 ):
    """
    Return a copy of the image with predicted template locations as bounding boxes overlaid on the image.
    
    The name of the template can also be displayed on top of the bounding box with showLabel=True
    Parameters
    ----------
    - image  : image in which the search was performed
    - tableHit: list of hit as returned by matchTemplates or findMatches
    - boxThickness: int
                    thickness of bounding box contour in pixels
    - boxColor: (int, int, int)
                RGB color for the bounding box
    - showLabel: Boolean
                Display label of the bounding box (field TemplateName)
    - labelColor: (int, int, int)
                RGB color for the label
    
    Returns
    -------
    outImage: RGB image
            original image with predicted template locations depicted as bounding boxes  
    """
    # Convert Grayscale to RGB to be able to see the color bboxes
    if image.ndim == 2: outImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert to RGB to be able to show detections as color box on grayscale image
    else:               outImage = image.copy()
        
    for index, row in tableHit.iterrows():
        x,y,w,h = row['BBox']
        cv2.rectangle(outImage, (x, y), (x+w, y+h), color=boxColor, thickness=boxThickness)
        if showLabel: cv2.putText(outImage, text=row['TemplateName'], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=labelScale, color=labelColor, lineType=cv2.LINE_AA) 
    
    return outImage


def drawBoxesOnGray(image, tableHit, boxThickness=2, boxColor=255, showLabel=False, labelColor=255, labelScale=0.5):
    """
    Same as drawBoxesOnRGB but with Graylevel.
    
    If a RGB image is provided, the output image will be a grayscale image
    
    Parameters
    ----------
    - image  : image in which the search was performed
    - tableHit: list of hit as returned by matchTemplates or findMatches
    - boxThickness: int
                thickness of bounding box contour in pixels
    - boxColor: int
                Gray level for the bounding box
    - showLabel: Boolean
                Display label of the bounding box (field TemplateName)
    - labelColor: int
                Gray level for the label
    
    Returns
    -------
    outImage: Single channel grayscale image
            original image with predicted template locations depicted as bounding boxes
    """
    # Convert RGB to grayscale
    if image.ndim == 3: outImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to RGB to be able to show detections as color box on grayscale image
    else:               outImage = image.copy()
        
    for index, row in tableHit.iterrows():
        x,y,w,h = row['BBox']
        cv2.rectangle(outImage, (x, y), (x+w, y+h), color=boxColor, thickness=boxThickness)
        if showLabel: cv2.putText(outImage, text=row['TemplateName'], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=labelScale, color=labelColor, lineType=cv2.LINE_AA) 
    
    return outImage