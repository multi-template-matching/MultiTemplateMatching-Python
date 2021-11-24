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


def NMS(tableHit, scoreThreshold=0.5, sortAscending=False, N_object=float("inf"), maxOverlap=0.5):
    """
    Perform Non-Maxima Supression (NMS).
    
    it compares the hits after maxima/minima detection, and removes detections overlapping above the maxOverlap  
    Also removes detections that do not satisfy a minimum score
   
    INPUT
    - tableHit         : Pandas DataFrame
                        List of potential detections as returned by MTM.findMatches.
                        Each row is a hit, with columns "TemplateName"(String),"BBox"(x,y,width,height),"Score"(float).
                        
    - scoreThreshold : Float, used to remove low score detections. 
                       If sortAscending=False, ie when best detections have high scores (correlation method), the detections with score below the threshold are discarded.
                       If sortAscending=True, ie when best detections have low scores (difference method), the detections with score above the threshold are discarded.
    
    - sortAscending : Boolean
                      use True when low score means better prediction (Difference-based score), False otherwise (Correlation score).
    
    - N_object      : int or infinity/float("inf") 
                      maximum number of best detections to return, to use when the number of object is known. 
                      Otherwise Default=infinity, ie return all detections passing NMS.
    
    - maxOverlap    : float between 0 and 1
                      the maximal overlap (IoU: Intersection over Union of the areas) authorised between 2 bounding boxes. 
                      Above this value, the bounding box of lower score is deleted.
    

    Returns
    -------
    Panda DataFrame with best detection after NMS, it contains max N detections (but potentially less)
    """
    listBoxes  = tableHit["BBox"].to_list()
    listScores = tableHit["Score"].to_list()
    
    if N_object==1:
        
        # Get row with highest or lower score
        if sortAscending:
            outTable = tableHit[tableHit.Score == tableHit.Score.min()]
        else:
            outTable = tableHit[tableHit.Score == tableHit.Score.max()]
        
        return outTable
        
    
    # N object > 1 -> do NMS
    if sortAscending: # invert score to have always high-score for bets prediction 
        listScores = [1-score for score in listScores] # NMS expect high-score for good predictions
        scoreThreshold = 1-scoreThreshold
    
    # Do NMS
    indexes = cv2.dnn.NMSBoxes(listBoxes, listScores, scoreThreshold, maxOverlap)
    
    # Get N best hit
    if N_object != float("inf"): 
        indexes = indexes[:N_object] 
        
    outTable = tableHit.iloc[indexes]
    
    return outTable

            
if __name__ == "__main__":
    import pandas as pd
    listHit =[ 
            {'TemplateName':1,'BBox':(780, 350, 700, 480), 'Score':0.8},
            {'TemplateName':1,'BBox':(806, 416, 716, 442), 'Score':0.6},
            {'TemplateName':1,'BBox':(1074, 530, 680, 390), 'Score':0.4}
            ]

    finalHits = NMS( pd.DataFrame(listHit), scoreThreshold=0.61, sortAscending=False, maxOverlap=0.8, N_object=1 )

    print(finalHits)
