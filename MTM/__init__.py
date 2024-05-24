"""Main code for Multi-Template-Matching (MTM)."""
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Sequence, Optional
from numpy.typing import NDArray

import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.feature import peak_local_max

from .NMS import NMS, Hit
from .version import __version__

__all__ = ['NMS']

# Define custom "types" for type hints
BBox = Tuple[int, int, int, int]  # bounding box in the form (x,y,width,height) with x,y top left corner
TemplateTuple = Tuple[str, NDArray, Optional[NDArray]]

def _findLocalMax_(corrMap:NDArray, score_threshold=0.6):
    """Get coordinates of the local maximas with values above a threshold in the image of the correlation map."""
    # If depending on the shape of the correlation map
    if corrMap.shape == (1,1): ## Template size = Image size -> Correlation map is a single digit')

        if corrMap[0,0] >= score_threshold:
            peaks = np.array([[0,0]])
        else:
            peaks = []

    # use scipy findpeaks for the 1D cases (would allow to specify the relative threshold for the score directly here rather than in the NMS
    elif corrMap.shape[0] == 1:     ## Template is as high as the image, the correlation map is a 1D-array
        peaks = find_peaks(corrMap[0], height=score_threshold) # corrMap[0] to have a proper 1D-array
        peaks = [[0,i] for i in peaks[0]] # 0,i since one coordinate is fixed (the one for which Template = Image)


    elif corrMap.shape[1] == 1: ## Template is as wide as the image, the correlation map is a 1D-array
        #peaks    = argrelmax(corrMap, mode="wrap")
        peaks = find_peaks(corrMap[:,0], height=score_threshold)
        peaks = [[i,0] for i in peaks[0]]


    else: # Correlation map is 2D
        peaks = peak_local_max(corrMap, threshold_abs=score_threshold, exclude_border=False).tolist()

    return peaks



def _findLocalMin_(corrMap:NDArray, score_threshold=0.4):
    """Find coordinates of local minimas with values below a threshold in the image of the correlation map."""
    return _findLocalMax_(-corrMap, -score_threshold)


def computeScoreMap(template:NDArray, image:NDArray, method:int = cv2.TM_CCOEFF_NORMED, mask=None):
    """
    Compute score map provided numpy array for template and image (automatically converts images if necessary).
    The template must be smaller or as large as the image.
    A mask can be provided to limit the comparison of the pixel values to a fraction of the template region.
    The mask should have the same dimensions and image type than the template.

    Return
    ------
    score map as numpy array
    """
    if template.dtype == "float64" or image.dtype == "float64":
        raise ValueError("64-bit images not supported, max 32-bit")

    # Convert images if not both 8-bit (OpenCV matchTemplate is only defined for 8-bit OR 32-bit)
    if not (template.dtype == "uint8" and image.dtype == "uint8"):
        template = np.float32(template)
        image    = np.float32(image)
        if mask is not None: mask = np.float32(mask)

    if mask is not None:

        if method not in (0,3):
           mask = None
           warnings.warn("Template matching method not compatible with use of mask (only 0/TM_SQDIFF or 3/TM_CCORR_NORMED).\n-> Ignoring mask.")

        else: # correct method
           # Check that mask has the same dimensions and type than template
            sameDimension = mask.shape == template.shape
            sameType = mask.dtype == template.dtype
            if not (sameDimension and sameType):
                mask = None
                warnings.warn("Mask does not have the same dimension or bit depth than the template.\n-> Ignoring mask.")


    # Compute correlation map
    return cv2.matchTemplate(image, template, method, mask=mask)


def findMatches(listTemplates : Sequence[TemplateTuple], image:NDArray, method:int = cv2.TM_CCOEFF_NORMED, N_object=float("inf"), score_threshold:float=0.5, searchBox:Optional[BBox] = None) -> List[Hit]:
    """
    Find all possible templates locations satisfying the score threshold provided a list of templates to search and an image.

    Parameters
    ----------
    - listTemplates : list of tuples (LabelString, template, mask (optional))
                      templates to search in each image, associated to a label
                      labelstring : string
                      template    : numpy array (grayscale or RGB)
                      mask (optional): numpy array, should have the same dimensions and type than the template

    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates

    - method : int
                one of OpenCV template matching method (0 to 5), default 5=0-mean cross-correlation

    - N_object: int or float("inf")
                expected number of objects in the image, default to infinity if unknown

    - score_threshold: float in range [0,1]
                if N_object>1, returns local minima/maxima respectively below/above the score_threshold

    - searchBox : tuple (x, y, width, height) in pixel unit
                optional rectangular search region as a tuple

    Returns
    -------
    A list of hit where each hit is a tuple as following ["TemplateName", (x, y, width, height), score]
    where template name is the name (or label) of the matching template
    (x, y, width, height) is a tuple of the bounding box coordinates in pixels, with xy the coordinates for the top left corner
    score (float) for the confidence of the detection
    """
    if N_object != float("inf") and not isinstance(N_object, int):
        raise TypeError("N_object must be an integer")
    
    ## Check image has not 0-width or height
    if image.shape[0] == 0:
        raise ValueError("Image has a height of 0.")
    
    if image.shape[1] == 0:
        raise ValueError("Image has a width of 0.")
    
    ## Crop image to search region if provided
    if searchBox is not None:
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset : yOffset + searchHeight, xOffset : xOffset + searchWidth]
    else:
        xOffset = yOffset = 0

    # Check that the template are all smaller are equal to the image (original, or cropped if there is a search region)
    for index, tempTuple in enumerate(listTemplates):

        if not isinstance(tempTuple, tuple) or len(tempTuple)<2:
            raise ValueError("listTemplates should be a list of tuples as ('name','array') or ('name', 'array', 'mask')")
        
        tempName = tempTuple[0]
        tempImage = tempTuple[1]
        
        # Check that template is not 0-width or height
        if tempImage.shape[0] == 0:
            raise ValueError(f"Template '{tempName}' has a height of 0.")
        
        if tempImage.shape[1] == 0:
            raise ValueError(f"Template '{tempName}' has a width of 0.")
        
        
        templateSmallerThanImage = all(templateDim <= imageDim for templateDim, imageDim in zip(tempImage.shape, image.shape)) # check both width/height smaller than image or search region

        if not templateSmallerThanImage :
            fitIn = "searchBox" if (searchBox is not None) else "image"
            raise ValueError("Template '{}' at index {} in the list of templates is larger than {}.".format(tempName, index, fitIn) )

    listHit = []
    # Use multi-threading to iterate through all templates, using half the number of cpu cores available.
    # i.e parallelize the search with the individual templates, in the same image
    with ThreadPoolExecutor(max_workers=round(os.cpu_count()*.5)) as executor:
        futures = [executor.submit(_multi_compute, tempTuple, image, method, N_object, score_threshold, xOffset, yOffset, listHit) for tempTuple in listTemplates]
        for future in as_completed(futures):
            _ = future.result()

    return listHit # All possible hits before Non-Maxima Supression

def _multi_compute(tempTuple : Sequence[TemplateTuple], image:NDArray, method:int, N_object:int, score_threshold:float, xOffset:int, yOffset:int, listHit:Sequence[Hit]):
    """
    Find all possible template locations satisfying the score threshold provided a template to search and an image.
    Add the hits found to the provided listHit, this function is running in parallel each instance for a different templates.

    Parameters
    ----------
    - tempTuple : a tuple (LabelString, template, mask (optional))
                      template to search in each image, associated to a label
                      labelstring : string
                      template    : numpy array (grayscale or RGB)
                      mask (optional): numpy array, should have the same dimensions and type than the template

    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates

    - method : int
                one of OpenCV template matching method (0 to 5), default 5=0-mean cross-correlation

    - N_object: int or float("inf")
                expected number of objects in the image, default to infinity if unknown

    - score_threshold: float in range [0,1]
                if N_object>1, returns local minima/maxima respectively below/above the score_threshold

    - xOffset : int 
                optional the x offset if the search area is provided

    - yOffset : int
                optional the y offset if the search area is provided

    - listHit : New hits are added to this list
    """
    templateName, template = tempTuple[:2]
    mask = None

    if len(tempTuple)>=3: # ie a mask is also provided
        if method in (0,3):
            mask = tempTuple[2]
        else:
            warnings.warn("Template matching method not supporting the use of Mask. Use 0/TM_SQDIFF or 3/TM_CCORR_NORMED.")

    #print('\nSearch with template : ',templateName)
    corrMap = computeScoreMap(template, image, method, mask=mask)

    ## Find possible location of the object
    if N_object==1: # Detect global Min/Max
        _, _, minLoc, maxLoc = cv2.minMaxLoc(corrMap)
        if method in (0,1):
            peaks = [minLoc[::-1]] # opposite sorting than in the multiple detection
        else:
            peaks = [maxLoc[::-1]]
    else:# Detect local max or min
        if method in (0,1): # Difference => look for local minima
            peaks = _findLocalMin_(corrMap, score_threshold)
        else:
            peaks = _findLocalMax_(corrMap, score_threshold)

    #print('Initially found',len(peaks),'hit with this template')
    height, width = template.shape[0:2] # slicing make sure it works for RGB too

    # For each peak create a hit as a list [templateName, (x,y,width,height), score] and add this hit into a bigger list
    newHits = [ (templateName, (int(peak[1]) + xOffset, int(peak[0]) + yOffset, width, height), corrMap[tuple(peak)] ) for peak in peaks]

    # Finally add these new hits to the original list of hits
    listHit.extend(newHits)


def matchTemplates(listTemplates:List[TemplateTuple], image:NDArray, method:int = cv2.TM_CCOEFF_NORMED, N_object = float("inf"), score_threshold:float = 0.5, maxOverlap:float = 0.25, searchBox:Optional[BBox] = None) -> List[Hit]:
    """
    Search each template in the image, and return the best N_object locations which offer the best score and which do not overlap above the maxOverlap threshold.

    Parameters
    ----------
    - listTemplates : list of tuples as (LabelString, template, mask (optional))
                      templates to search in each image, associated to a label
                      labelstring : string
                      template    : numpy array (grayscale or RGB)
                      mask (optional): numpy array, should have the same dimensions and type than the template

    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates

    - method : int
               one of OpenCV template matching method (1 to 5), default 5=0-mean cross-correlation
               method 0 is not supported (no NMS implemented for non-bound difference score), use 1 instead

    - N_object: int or foat("inf")
                expected number of objects in the image, default to infinity if unknown

    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold

    - maxOverlap: float in range [0,1]
                This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
                If the ratio is over the maxOverlap, the lower score bounding box is discarded.

    - searchBox : tuple (X, Y, Width, Height) in pixel unit
                optional rectangular search region as a tuple

    Returns
    -------
    A list of hit, where each hit is a tuple in the form (label, (x, y, width, height), score)
        if N=1, return the best matches independently of the score_threshold
        if N<inf, returns up to N best matches that passed the NMS
        if N=inf, returns all matches that passed the NMS
    """
    if maxOverlap < 0 or maxOverlap > 1:
        raise ValueError("Maximal overlap between bounding box is in range [0-1]")

    listHits = findMatches(listTemplates, image, method, N_object, score_threshold, searchBox)

    if method == 0: 
        raise ValueError("The method TM_SQDIFF is not supported. Use TM_SQDIFF_NORMED instead.")
    
    sortAscending = (method==1)

    return NMS(listHits, score_threshold, sortAscending, N_object, maxOverlap)


def drawBoxesOnRGB(image:NDArray, listHit:Sequence[Hit], boxThickness:int=2, boxColor:Tuple[int,int,int] = (255, 255, 00), showLabel:bool=False, labelColor=(255, 255, 0), labelScale=0.5) -> NDArray:
    """
    Return a copy of the image with predicted template locations as bounding boxes overlaid on the image
    The name of the template can also be displayed on top of the bounding box with showLabel=True

    Parameters
    ----------
    - image  : image in which the search was performed

    - listHit: list of hit as returned by matchTemplates or findMatches

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
    outImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy

    for label, bbox, _ in listHit:
        
        x,y,w,h = bbox
        cv2.rectangle(outImage, (x, y), (x + w, y + h), color = boxColor, thickness = boxThickness)
        
        if showLabel: 
            cv2.putText(outImage, 
                        text=label, 
                        org=(x, y), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=labelScale, 
                        color=labelColor, 
                        lineType=cv2.LINE_AA)

    return outImage


def drawBoxesOnGray(image:NDArray, listHit:Sequence[Hit], boxThickness=2, boxColor=255, showLabel=False, labelColor=255, labelScale=0.5) -> NDArray:
    """
    Same as drawBoxesOnRGB but with Graylevel.
    If a RGB image is provided, the output image will be a grayscale image

    Parameters
    ----------
    - image  : image in which the search was performed

    - listHit: list of hit as returned by matchTemplates or findMatches

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
    outImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

    for label, bbox, _ in listHit:
        
        x,y,w,h = bbox
        cv2.rectangle(outImage, (x, y), (x+w, y+h), color=boxColor, thickness=boxThickness)
        
        if showLabel: 
            cv2.putText(outImage, 
                        text=label, 
                        org=(x, y), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=labelScale, 
                        color=labelColor, 
                        lineType=cv2.LINE_AA)

    return outImage