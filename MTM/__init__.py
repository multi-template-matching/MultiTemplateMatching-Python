import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy.signal	 import find_peaks

from MTM.NMS import NMS

__all__ = ['NMS']


def _findLocalMax_(corrMap, score_threshold=0.6):
	'''
	Get coordinates of the local maximas with values above a threshold in the image of the correlation map
	'''
	
	# IF depending on the shape of the correlation map
	if corrMap.shape == (1,1): ## Template size = Image size -> Correlation map is a single digit')
		
		if corrMap[0,0]>=score_threshold:
			Peaks = np.array([[0,0]])
		else:
			Peaks = []

	# use scipy findpeaks for the 1D cases (would allow to specify the relative threshold for the score directly here rather than in the NMS
	elif corrMap.shape[0] == 1:		## Template is as high as the image, the correlation map is a 1D-array
		Peaks = find_peaks(corrMap[0], height=score_threshold) # corrMap[0] to have a proper 1D-array
		Peaks = [[0,i] for i in Peaks[0]] # 0,i since one coordinate is fixed (the one for which Template = Image)
		

	elif corrMap.shape[1] == 1: ## Template is as wide as the image, the correlation map is a 1D-array
		#Peaks	  = argrelmax(corrMap, mode="wrap")
		Peaks = find_peaks(corrMap[:,0], height=score_threshold)
		Peaks = [[i,0] for i in Peaks[0]]


	else: # Correlatin map is 2D
		Peaks = peak_local_max(corrMap, threshold_abs=score_threshold, exclude_border=False).tolist()

	return Peaks



def _findLocalMin_(corrMap, score_threshold=0.4):
	'''Find coordinates of local minimas with values below a threshold in the image of the correlation map'''
	return _findLocalMax_(-corrMap, -score_threshold)



def findMatches(listTemplates, image, method=cv2.TM_CCOEFF_NORMED, N_object=float("inf"), score_threshold=0.5, searchBox=None):
	'''
	Find all possible templates locations provided a list of template to search and an image
	- listTemplate : list of tuples [(templateName, templateImage), (templateName2, templateImage2) ]
	- method : one of OpenCV template matching method (0 to 5)
	- N_object: expected number of object in the image
	- score_threshold: if N>1, returns local minima/maxima respectively below/above the score_threshold
	- searchBox : optional search region as a tuple (X, Y, Width, Height)
	'''
	if N_object!=float("inf") and type(N_object)!=int:
		raise TypeError("N_object must be an integer")
		
	elif N_object<1:
		raise ValueError("At least one object should be expected in the image")
		
	## Crop image to search region if provided
	if searchBox != None: 
		xOffset, yOffset, searchWidth, searchHeight = searchBox
		image = image[yOffset:yOffset+searchHeight, xOffset:xOffset+searchWidth]
	else:
		xOffset=yOffset=0
	
	## 16-bit image are converted to 32-bit for matchTemplate
	if image.dtype == 'uint16': image = np.float32(image)	
	
	listHit = []
	for templateName, template in listTemplates:
		
		#print('\nSearch with template : ',templateName)
		## 16-bit image are converted to 32-bit for matchTemplate
		if template.dtype == 'uint16': template = np.float32(template)		  
		
		## Compute correlation map
		corrMap = cv2.matchTemplate(template, image, method)

		## Find possible location of the object 
		if N_object==1: # Detect global Min/Max
			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(corrMap)
			
			if method==1:
				Peaks = [minLoc[::-1]] # opposite sorting than in the multiple detection
			
			elif method in (3,5):
				Peaks = [maxLoc[::-1]]
			
			
		else:# Detect local max or min
			if method==1: # Difference => look for local minima
				Peaks = _findLocalMin_(corrMap, score_threshold)
			
			elif method in (3,5):
				Peaks = _findLocalMax_(corrMap, score_threshold)
			
		
		#print('Initially found',len(Peaks),'hit with this template')
		
		
		# Once every peak was detected for this given template
		## Create a dictionnary for each hit with {'TemplateName':, 'BBox': (x,y,Width, Height), 'Score':coeff}
		
		height, width = template.shape[0:2] # slicing make sure it works for RGB too
		
		for peak in Peaks :
			coeff  = corrMap[tuple(peak)]
			newHit = {'TemplateName':templateName, 'BBox': [int(peak[1])+xOffset, int(peak[0])+yOffset, width, height], 'Score':coeff}

			# append to list of potential hit before Non maxima suppression
			listHit.append(newHit)
	
	
	return listHit # All possible hit before Non-Maxima Supression
	

def matchTemplates(listTemplates, image, method=cv2.TM_CCOEFF_NORMED, N_object=float("inf"), score_threshold=0.5, maxOverlap=0.25, searchBox=None):
	'''
	Search each template in the image, and return the best N_object location which offer the best score and which do not overlap
	- listTemplate : list of tuples (templateName, templateImage)
	- method : one of OpenCV template matching method (0 to 5)
	- N_object: expected number of object in the image
	- score_threshold: if N>1, returns local minima/maxima respectively below/above the score_threshold
	'''
	if maxOverlap<0 or maxOverlap>1:
		raise ValueError("Maximal overlap between bounding box is in range [0-1]")
		
	listHit = findMatches(listTemplates, image, method, N_object, score_threshold, searchBox)
	
	if method == 1:		  bestHits = NMS(listHit, N_object=N_object, maxOverlap=maxOverlap, sortDescending=False)
	
	elif method in (3,5): bestHits = NMS(listHit, N_object=N_object, maxOverlap=maxOverlap, sortDescending=True)
	
	return bestHits


def drawBoxes(img, listHit, boxThickness=2, boxColor=(255, 255, 00), showLabel=True, labelColor=(255, 255, 0) ):
	"""
	Return a copy of the image with results of template matching drawn as yellow rectangle and name of the template on top
	"""
	outImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert to RGB to be able to show detections as color box on grayscale image
	
	for hit in listHit:
		x,y,w,h = hit['BBox']
		cv2.rectangle(outImage, (x, y), (x+w, y+h), color=boxColor, thickness=boxThickness)
		if showLabel: cv2.putText(outImage, text=hit['TemplateName'], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=labelColor, lineType=cv2.LINE_AA) 
	
	return outImage


if __name__ == '__main__':
	
	from skimage.data import coins
	import matplotlib.pyplot as plt
	
	## Get image and template
	smallCoin = coins()[37:37+38, 80:80+41] 
	bigCoin	  = coins()[14:14+59,302:302+65]
	image = coins()
	
	## Perform matching
	listHit = matchTemplates([('small', smallCoin), ('big', bigCoin)], image, score_threshold=0.4, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0)
	#listHit = matchTemplates([('small', smallCoin), ('big', bigCoin)], image, N_object=1, score_threshold=0.4, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0)
   
	print("Found {} coins".format(len(listHit)))
	
	for hit in listHit:
		print(hit)
	
	## Display matches
	Overlay = drawBoxes(image, listHit)
	plt.imshow(Overlay)
