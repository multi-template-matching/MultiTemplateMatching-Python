# -*- coding: utf-8 -*-
"""
Non-Maxima Supression (NMS) for match template.

From a pool of bounding box each predicting possible object locations with a given score, 
the NMS removes the bounding boxes overlapping with a bounding box of higher score above the maxOverlap threshold

This effectively removes redundant detections of the same object and still allow the detection of close objects (ie small possible overlap)
The "possible" allowed overlap is set by the variable maxOverlap (between 0 and 1) which is the ratio Intersection over Union (IoU) area for a given pair of overlaping BBoxes

Since version 1.6.0 the NMS is assured by opencv cv2.dnn.NMSBoxes function

@author: Laurent Thomas
"""
import cv2
from typing import Tuple, List, Sequence

Hit = Tuple[str, Tuple[int, int, int, int], float]

def NMS(listHit:Sequence[Hit], scoreThreshold:float=0.5, sortAscending:bool=False, N_object=float("inf"), maxOverlap:float=0.5) -> List[Hit]:
    """
    Perform Non-Maxima Supression (NMS)
    
    it compares the hits after maxima/minima detection, and removes detections overlapping above the maxOverlap  
    Also removes detections that do not satisfy a minimum score
   
    Parameters
    ----------
    - listHit
    A list (or equivalent) of potential detections in the form (label, (x,y,width,height), score) as returned by MTM.findMatches.
                        
    - scoreThreshold, float, default 0.5
    used to remove low score detections. 
    If sortAscending=False, ie when best detections have high scores (correlation method), the detections with score below the threshold are discarded.
    If sortAscending=True, ie when best detections have low scores (difference method), the detections with score above the threshold are discarded.
    
    - sortAscending, bool
    use True when low score means better prediction (Difference-based score), False otherwise (Correlation score).
    
    - N_object, int or infinity/float("inf") 
    maximum number of best detections to return, to use when the number of object is known. 
    Otherwise Default=infinity, ie return all detections passing NMS.
    
    - maxOverlap, float between 0 and 1
    the maximal overlap (IoU: Intersection over Union of the areas) authorised between 2 bounding boxes. 
    Above this value, the bounding box of lower score is deleted.
    

    Returns
    -------
    A list of hit with no overlapping detection (not above the maxOverlap)
    """
    nHits = len(listHit)
    if nHits <= 1:
        return listHit[:] # same just making a copy to avoid side effects

    # Get separate lists for the bounding boxe coordinates and their score
    listBoxes  = [None] * nHits # list of (x,y,width,height)
    listScores = [None] * nHits  # list of associated scores
    
    for i, hit in enumerate(listHit): # single iteration through the list instead of using 2 list comprehensions
        listBoxes[i] = hit[1]
        listScores[i] = hit[2] 
    
    if N_object == 1:
        
        # Get hit with highest or lower score
        if sortAscending:
            bestHit = min(listHit, key = lambda hit: hit[2])
        else:
            bestHit = max(listHit, key = lambda hit: hit[2])
        
        return [bestHit] # wrap it into a list so the function always returns a list
        
    
    # N object > 1 -> do NMS
    if sortAscending: # invert score to have always high-score for bets prediction 
        listScores = [1-score for score in listScores] # NMS expect high-score for good predictions
        scoreThreshold = 1-scoreThreshold
    
    # Do NMS, it returns a list of the positional indexes of the hits in the original list that pass the NMS test
    indexes = cv2.dnn.NMSBoxes(listBoxes, listScores, scoreThreshold, maxOverlap)
    
    # Eventually take only up to n hit if provided
    if N_object != float("inf"): 
        indexes = indexes[:N_object] 
        
    return [listHit[x] for x in indexes]
            
if __name__ == "__main__":
    import numpy as np

    listHit = [("1", (780, 350, 700, 480),  0.8),
               ("1", (806, 416, 716, 442),  0.6),
               ("1", (1074, 530, 680, 390), 0.4)]
            

    finalHits = NMS(listHit, scoreThreshold=0.3, sortAscending=False, maxOverlap=0.5, N_object=2)

    print(np.array(finalHits, dtype=object)) # does not work if not specifying explicitely dtype=object
