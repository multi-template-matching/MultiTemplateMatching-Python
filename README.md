[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/multi-template-matching/MultiTemplateMatching-Python/master?filepath=tutorials)
![Twitter Follow](https://img.shields.io/twitter/follow/LauLauThom?style=social)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/laurent132.thomas@laposte.net)

# Multi-Template-Matching
Multi-Template-Matching is a python package to perform object-recognition in images using one or several smaller template images.  
The main function `MTM.matchTemplates` returns the best predicted locations provided either a score_threshold and/or the expected number of objects in the image.  

The branch opencl contains some test using the UMat object to run on GPU, but it is actually slow, which can be expected for small dataset as the transfer of the data between the CPU and GPU is slow.

# News
- Version 2.0.0 got rid of the dependency on pandas, the list of hit is simply a python list. The new version also features extensive type hints in functions
- 03/03/2023 : Version 1.6.4 contributed by @bartleboeuf comes with speed enhancement thanks to parallelizing of the individual template searches.
Thanks for this first PR !!
- 10/11/2021 : You might be interested to test the newer python implementation which is more object-oriented and only relying on scikit-image and shapely.*
https://github.com/multi-template-matching/mtm-python-oop 

# Installation
Using pip in a python environment, `pip install Multi-Template-Matching`  
Once installed, `import MTM`should work.  
Example jupyter notebooks can be downloaded from the tutorial folder of the github repository and executed in the newly configured python environement.  

## Install in dev mode
If you want to contribute or experiment with the source code, you can install the package "from source", by first downloading or cloning the repo.    
Then opening a command prompt in the repo's root directory (the one containing this README) and calling `pip install -e .` (mind the final dot).    
- the `-e` flag stands for editable and make sure that any change to the source code will be directly reflected when you import the package in your script  
- the . just tell pip to look for the package to install in the current directory

# Documentation
The [wiki](https://github.com/multi-template-matching/MultiTemplateMatching-Python/wiki) section of the repo contains a mini API documentation with description of the key functions of the package.   
The [website](https://multi-template-matching.github.io/Multi-Template-Matching/) of the project contains some more general documentation.

# Tips and tricks

- `matchTemplates` expects for the template a list of tuples with one tuple per template in the form (a string identifier for the template, array for the template image). You can generate such list of tuples, from separate lists of the label and template images as following

```python
listLabel    = []  # this one should have the string identifier for each template
listTemplate = [] # this one should have the image array for each template, both lists should have the same length
listTemplateTuple = zip(listLabel, listTemplate)
```

Similarly, from the list of hits returned by matchTemplates (or NMS), you can get individual lists for the label, bounding-boxes and scores, using `listLabel, listBbox, listScore = zip(*listHit)`

- To have a nicer formatting (one list item per row) when printing the list of detected hits, you can use the `pprint` function from the pprint module (for pretty print).  
It's usually not needed in notebooks (see the example notebooks).  
```
from pprint import pprint
pprint(listHit)
```

- Before version 2.0.0, most functions were returning or accepting pandas DataFrame for the list of hit. 
You can still get such DataFrame from the list of hits returned by MTM v2.0.0 and later, using the command below  

```python
import pandas as pd

listLabel, listBbox, listScore = zip(*listHit)

df = pd.DataFrame({"Label":listLabel,
				   "bounding box":listBbox,
				   "Score":listScore})

print(df)
```

You can also stick to previous version of MTM by specifying the version to pip install, or in a requirements.txt or environment.yml
`pip install "Multi-Template-Matching < 2.0.0"

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

Download the citation as a .ris file from the journal website, [here](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3363-7.ris).

# Releases
Previous github releases were archived to Zenodo, but the best is to use pip to install specific versions.  
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
