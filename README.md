[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/multi-template-matching/MultiTemplateMatching-Python/master?filepath=tutorials)
![Twitter Follow](https://img.shields.io/twitter/follow/LauLauThom?style=social)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/laurent132.thomas@laposte.net)

# Multi-Template-Matching
Multi-Template-Matching is a package to perform object-recognition in images using one or several smaller template images.  
The template and images should have the same bitdepth (8,16,32-bit) and number of channels (single/Grayscale or RGB).  
The main function `MTM.matchTemplates` returns the best predicted locations provided either a score_threshold and/or the expected number of objects in the image.  

# Installation
Using pip in a python environment, `pip install Multi-Template-Matching`  
Once installed, `import MTM`should work.  
Example jupyter notebooks can be downloaded from the tutorial folder of the github repository and executed in the newly configured python environement.  

# Documentation
The [wiki](https://github.com/multi-template-matching/MultiTemplateMatching-Python/wiki) section of the repo contains a mini API documentation with description of the key functions of the package.

# Examples
Check out the [jupyter notebook tutorial](https://github.com/multi-template-matching/MultiTemplateMatching-Python/tree/master/tutorials) for some example of how to use the package.  
You can run the tutorials online using Binder, no configuration needed ! (click the Binder banner on top of this page).  
To run the tutorials locally, install the package using pip as described above, then clone the repository and unzip it.  
Finally open a jupyter-notebook session in the unzipped folder to be able to open and execute the notebook.  
The [wiki](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji/wiki) section of this related repository also provides some information about the implementation.

# Citation
If you use this implementation for your research, please cite:
  
Thomas, L.S.V., Gehrig, J. Multi-template matching: a versatile tool for object-localization in microscopy images.  
BMC Bioinformatics 21, 44 (2020). https://doi.org/10.1186/s12859-020-3363-7

# Releases
Previous github releases were archived to Zenodo.  
[![DOI](https://zenodo.org/badge/197186256.svg)](https://zenodo.org/badge/latestdoi/197186256)

# Related projects
See this [repo](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji) for the implementation as a Fiji plugin.  
[Here](https://nodepit.com/workflow/com.nodepit.space%2Flthomas%2Fpublic%2FMulti-Template%20Matching.knwf) for a KNIME workflow using Multi-Template-Matching.


# Origin of the work
This work has been part of the PhD project of **Laurent Thomas** under supervision of **Dr. Jochen Gehrig** at ACQUIFER.  

<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/Acquifer_Logo_60k_cmyk_300dpi.png" alt="ACQUIFER" width="400" height="80">     

# Funding
This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement No 721537 ImageInLife.  

<p float="left">
<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/ImageInlife.png" alt="ImageInLife" width="130" height="100">
<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/MarieCurie.jpg" alt="MarieCurie" width="130" height="130">
</p>
