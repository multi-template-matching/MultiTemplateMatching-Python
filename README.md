# Multi-Template-Matching
Multi-Template-Matching is a package to perform object-recognition in images using one or several smaller template images.  
The template and images should have the same bitdepth (8,16,32-bit) and number of channels (single/Grayscale or RGB).  
The main function `MTM.matchTemplates` returns the best predicted locations provided either a score_threshold and/or the expected number of objects in the image.  

# Installation
Using pip in a python environment, `pip install Multi-Template-Matching`  
Once installed, `import MTM`should work.  
Example jupyter notebooks can be downloaded from the tutorial folder of the github repository and executed in the newly configured python environement.  

# Documentation
The package MTM contains mostly 2 important functions:  

## matchTemplates  
`matchTemplates(listTemplates, image, method=cv2.TM_CCOEFF_NORMED, N_object=float("inf"), score_threshold=0.5, maxOverlap=0.25, searchBox=None)`  

This function searches each template in the image, and return the best N_object location which offer the best scores and which do not overlap above the `maxOverlap` threshold.  

__Parameters__
- _listTemplates_:   
            list of tuples (LabelString, Grayscale or RGB numpy array) templates to search in each image, associated to a label 

- _image_  : Grayscale or RGB numpy array  
           image in which to perform the search, it should be the same bitDepth and number of channels than the templates

- _method_ : int   
            one of OpenCV template matching method (0 to 5), default 5=0-mean cross-correlation

- _N_object_: int  
            expected number of objects in the image

- score_threshold: float in range [0,1]  
            if N>1, returns local minima/maxima respectively below/above the score_threshold

- _maxOverlap_: float in range [0,1]  
            This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
            If the ratio is over the maxOverlap, the lower score bounding box is discarded.

- _searchBox_ : tuple (X, Y, Width, Height) in pixel unit  
            optional rectangular search region as a tuple
    
__Returns__
- Pandas DataFrame with 1 row per hit and column "TemplateName"(string), "BBox":(X, Y, Width, Height), "Score":float             
            - if N=1, return the best match independently of the score_threshold  
            - if N<inf, returns up to N best matches that passed the score_threshold  
            - if N=inf, returns all matches that passed the score_threshold  


The function `findMatches` performs the same detection without the Non-Maxima Supression.  

## drawBoxesOnRGB
The 2nd important function is `drawBoxesOnRGB` to display the detections as rectangular bounding boxes on the initial image.  
To be able to visualise the detection as colored bounding boxes, the function return a RGB copy of the image if a grayscale image is provided.  
It is also possible to draw the detection bounding boxes on the grayscale image using drawBoxesOnGray (for instance to generate a mask of the detections).      
`drawBoxesOnRGB(image, hits, boxThickness=2, boxColor=(255, 255, 00), showLabel=True, labelColor=(255, 255, 0), labelScale=0.5 )`

This function returns a copy of the image with predicted template locations as bounding boxes overlaid on the image
The name of the template can also be displayed on top of the bounding boxes with showLabel=True.

__Parameters__
- _image_  : numpy array  
        image in which the search was performed  
        
- _hits_ : pandas dataframe  
         hits as returned by matchTemplates or findMatches  
        
- _boxThickness_: int  
        thickness of bounding box contour in pixels. -1 will fill the bounding box (useful for masks).  
        
- _boxColor_: (int, int, int)  
        RGB color for the bounding box  

- _showLabel_: Boolean, default True  
        Display label of the bounding box (field TemplateName)

- _labelColor_: (int, int, int)  
          RGB color for the label  

- _labelScale_: float, default=0.5
	scale for the label sizes

__Returns__
- _outImage_: RGB image  
        original image with predicted template locations depicted as bounding boxes  

# Examples
Check out the [jupyter notebook tutorial](https://github.com/multi-template-matching/MultiTemplateMatching-Python/tree/master/tutorials) for some example of how to use the package.  
The [wiki](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji/wiki) section of this related repository also provides some information about the implementation.

# Citation
If you use this implementation for your research, please cite:
  
_Multi-Template Matching: a versatile tool for object-localization in microscopy images;_  
_Laurent SV Thomas, Jochen Gehrig_  
bioRxiv 619338; doi: https://doi.org/10.1101/619338

# Releases
New github releases are automatically archived to Zenodo.  
[![DOI](https://zenodo.org/badge/197186256.svg)](https://zenodo.org/badge/latestdoi/197186256)

# Related projects
See this [repo](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji) for the implementation as a Fiji plugin.  
[Here](https://nodepit.com/workflow/com.nodepit.space%2Flthomas%2Fpublic%2FMulti-Template%20Matching.knwf) for a KNIME workflow using Multi-Template-Matching.


# Origin of the work
This work has been part of the PhD project of **Laurent Thomas** under supervision of **Dr. Jochen Gehrig** at:  
  
ACQUIFER a division of DITABIS AG  
Digital Biomedical Imaging Systems AG  
Freiburger Str. 3  
75179 Pforzheim  

<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/Acquifer_Logo_60k_cmyk_300dpi.png" alt="Fish" width="400" height="80">     

# Funding
This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement No 721537 ImageInLife.  

<p float="left">
<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/ImageInlife.png" alt="ImageInLife" width="130" height="100">
<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/MarieCurie.jpg" alt="MarieCurie" width="130" height="130">
</p>
