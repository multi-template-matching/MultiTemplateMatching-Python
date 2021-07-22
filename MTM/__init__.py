"""
Multi-Template-Matching.
Implements object-detection with one or mulitple template images
Detected locations are represented as bounding boxes.
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.lines import Line2D
from skimage import feature, transform
from .NMS import NMS
from .Detection import BoundingBox

__all__ = ['NMS']
__version__ = '1.5.4'


def findMaximas(corrMap, score_threshold=0.6, nObjects=float("inf")):
    """
    Maxima detection in correlation map.
    Get coordinates of the global (nObjects=1)
    or local maximas with values above a threshold
    in the image of the correlation map.
    """
    # IF depending on the shape of the correlation map
    if corrMap.shape == (1, 1):  # Template size = Image size -> Correlation map is a single digit representing the score
        listPeaks = np.array([[0, 0]]) if corrMap[0, 0] >= score_threshold else []

    else:  # Correlation map is a 1D or 2D array
        nPeaks = 1 if nObjects == 1 else float("inf")  # global maxima detection if nObject=1 (find best hit of the score map)
        # otherwise local maxima detection (ie find all peaks), DONT LIMIT to nObjects, more than nObjects detections might be needed for NMS
        listPeaks = feature.peak_local_max(corrMap,
                                           threshold_abs=score_threshold,
                                           exclude_border=False,
                                           num_peaks=nPeaks).tolist()

    return listPeaks


def findMatches(image,
                listTemplates,
                listLabels=None,
                score_threshold=0.5,
                nObjects=float("inf"),
                searchBox=None,
                downscaling_factor=1):
    """
    Find all possible templates locations provided a list of template to search and an image.
    Parameters
    ----------
    - image  : Grayscale or RGB numpy array
              image in which to perform the search, it should be the same bitDepth and number of channels than the templates
    - listTemplates : list of templates as grayscale or RGB numpy array
                      templates to search in each image
    - listLabels (optional) : list of string labels associated to the templates (order must match).
                              these labels can describe categories associated to the templates
    - nObjects: int
                expected number of objects in the image
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    - searchBox (optional): tuple (x y, width, height) in pixels
                limit the search to a rectangular sub-region of the image
    - downscaling_factor: int >= 1, default 1 (ie no downscaling)
               speed up the search by downscaling the template and image size before running the template matching
               detected regions are then rescaled to original image sizes.
    Returns
    -------
    - List of Detections
    """
    if nObjects != float("inf") and type(nObjects) != int:
        raise TypeError("nObjects must be an integer")

    if nObjects < 1:
        raise ValueError("At least one object should be expected in the image")

    if (listLabels is not None and
       (len(listTemplates) != len(listLabels))):
        raise ValueError("There must be one label per template.")

    if downscaling_factor < 1:
        raise ValueError("Downscaling factor must be an integer over 1")

    # Crop image to search region if provided
    if searchBox is not None:
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset:yOffset+searchHeight,
                      xOffset:xOffset+searchWidth]
    else:
        xOffset = yOffset = 0
    
    # TODO dont change listTemplate in-place + use a list comprehension
    if downscaling_factor != 1:
        image = transform.rescale(image, 1/downscaling_factor, anti_aliasing = False)
        listTemplates = [transform.rescale(template, 1/downscaling_factor, anti_aliasing = False) for template in listTemplates]

    listHit = []
    for index, template in enumerate(listTemplates):

        corrMap = feature.match_template(image, template)
        listPeaks = findMaximas(corrMap, score_threshold, nObjects)

        height, width = template.shape[0:2]  # slicing make sure it works for RGB too
        label = listLabels[index] if listLabels else ""

        for peak in listPeaks:
            score = corrMap[tuple(peak)]
            bbox = (int(peak[1]) * downscaling_factor + xOffset,
                    int(peak[0]) * downscaling_factor + yOffset,
                    width * downscaling_factor, height * downscaling_factor)

            hit = BoundingBox(bbox, score, index, label)
            listHit.append(hit)  # append to list of potential hit before Non maxima suppression

    return listHit  # All possible hits before Non-Maxima Supression


def matchTemplates(image,
                   listTemplates,
                   listLabels=None,
                   score_threshold=0.5,
                   maxOverlap=0.25,
                   nObjects=float("inf"),
                   searchBox=None,
                   downscaling_factor=1):
    """
    Search each template in the image, and return the best nObjects locations which offer the best score and which do not overlap.
    Parameters
    ----------
    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates
               
    - listTemplates : list of templates as 2D grayscale or RGB numpy array
                      templates to search in each image
    
    - listLabels (optional) : list of string
                              labels, associated the templates. The order of the label must match the template order.
    
    - nObjects: int
                expected number of objects in the image
    
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    
    - maxOverlap: float in range [0,1]
                This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
                If the ratio is over the maxOverlap, the lower score bounding box is discarded.
    
    - searchBox : tuple (x y, width, height) in pixels
                limit the search to a rectangular sub-region of the image
                
    - downscaling_factor: int >= 1
               speed up the search by downscaling the template and image size before running the template matching
               detected regions are then rescaled to original image sizes.
               
    Returns
    -------
    List with 1 element per hit and each element containing "Score"(float), "BBox"(X, Y, X, Y), "template_index"(int), "Label"(string)
        if nObjects=1, return the best matches independently of the score_threshold
        if nObjects<inf, returns up to N best matches that passed the score_threshold
        if nObjects='inf'(string), returns all matches that passed the score_threshold
        
    """
    if maxOverlap<0 or maxOverlap>1:
        raise ValueError("Maximal overlap between bounding box is in range [0-1]")

    listHit  = findMatches(image, listTemplates, listLabels, score_threshold, nObjects, searchBox, downscaling_factor)
    bestHits = NMS(listHit, maxOverlap, nObjects)

    return bestHits


def plotDetections(image, listDetections, thickness=2, showLegend=False, showScore=False):
    """
    Plot the detections overlaid on the image.
    This generates a Matplotlib figure and displays it.
    Detections with identical template index (ie categories)
    are shown with identical colors.
    Parameters
    ----------
    - image  :
        image in which the search was performed
    - listDetections:
        list of detections as returned by matchTemplates or findMatches
    - thickness (optional, default=2): int
        thickness of plotted contour in pixels
    - showLegend (optional, default=False): Boolean
        Display a legend panel with the category labels for each color.
        This works if the Detections have a label
        (not just "", in which case the legend is not shown).
    - showScore (optional, default=False): Boolean
        Display the score of the corresponding hit next to a plotted contour.
    """
    plt.figure()
    plt.imshow(image, cmap="gray")  # cmap gray only impacts gray images
    # RGB are still displayed as color

    # Load a color palette for categorical coloring of detections
    # ie same category (identical tempalte index) = same color
    palette = plt.cm.Set3.colors
    nColors = len(palette)

    if showLegend:
        mapLabelColor = {}

    for detection in listDetections:

        # Get color for this category
        colorIndex = detection.get_template_index() % nColors  # will return an integer in the range of palette
        color = palette[colorIndex]

        plt.plot(*detection.get_lists_xy(),
                 linewidth=thickness,
                 color=color)

        if showScore:
            (x, y, width, height) = detection.get_xywh()
            plt.annotate(round(detection.get_score(), 2),
                         (x + width/3, y + height/3),
                         ha="center",
						 fontsize=height/4)

        # If show legend, get detection label and current color
        if showLegend:

            label = detection.get_label()

            if label != "":
                mapLabelColor[label] = color

    # Finally add the legend if mapLabelColor is not empty
    if showLegend :

        if not mapLabelColor:  # Empty label mapping
            warnings.warn("No label associated to the templates." +
                          "Skipping legend.")

        else:  # meaning mapLabelColor is not empty

            legendLabels = []
            legendEntries = []

            for label, color in mapLabelColor.items():
                legendLabels.append(label)
                legendEntries.append(Line2D([0], [0], color=color, lw=4))

            plt.legend(legendEntries, legendLabels)

def rescale_bounding_boxes(listDetectionsdownscale, downscaling_factor):
    """
    Rescale detected bounding boxes to the original image resolution, when downscaling was used for the detection.
    Parameters
    ----------
    - listDetections : list of BoundingBox items
        List with 1 element per hit and each element containing "Score"(float), "BBox"(X, Y, X, Y), "Template_index"(int), "Label"(string)
    - downscaling_factor: int >= 1
               allows to rescale by multiplying coordinates by the factor they were downscaled by
    Returns
    -------
    listDetectionsupscaled : list of BoundingBox items
        List with 1 element per hit and each element containing "Score"(float), "BBox"(X, Y, X, Y) (in coordinates of the full scale image), "Template_index"(int), "Label"(string)
    """
    listDetectionsupscale = []

    for detection in listDetectionsdownscale:

        (x, y, w, h), score, index, label = detection.get_xywh(), detection.get_score(), detection.get_template_index(), detection.get_label()

        bboxupscale = (x*downscaling_factor, y*downscaling_factor, w*downscaling_factor, h*downscaling_factor)
        detectionupscale = BoundingBox(bboxupscale, score, index, label)

        listDetectionsupscale.append(detectionupscale)

    return listDetectionsupscale            