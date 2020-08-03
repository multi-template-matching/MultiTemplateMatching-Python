# -*- coding: utf-8 -*-
"""
Non-Maxima Supression (NMS) for match template
From a pool of bounding box each predicting possible object locations with a given score, 
the NMS removes the bounding boxes overlapping with a bounding box of higher score above the maxOverlap threshold

This effectively removes redundant detections of the same object and still allow the detection of close objects (ie small possible overlap)
The "possible" allowed overlap is set by the variable maxOverlap (between 0 and 1) which is the ratio Intersection over Union (IoU) area for a given pair of overlaping BBoxes


@author: Laurent Thomas
"""

import cv2


def NMS(tableHit, scoreThreshold=0.5, sortAscending=False, N_object=float("inf"), maxOverlap=0.5):
    '''
    Perform Non-Maxima supression : it compares the hits after maxima/minima detection, and removes the ones that are too close (too large overlap)
    This function works both with an optionnal threshold on the score, and number of detected bbox

    if a scoreThreshold is specified, we first discard any hit below/above the threshold (depending on sortDescending)
    if sortDescending = True, the hit with score below the treshold are discarded (ie when high score means better prediction ex : Correlation)
    if sortDescending = False, the hit with score above the threshold are discared (ie when low score means better prediction ex : Distance measure)

    Then the hit are ordered so that we have the best hits first.
    Then we iterate over the list of hits, taking one hit at a time and checking for overlap with the previous validated hit (the Final Hit list is directly iniitialised with the first best hit as there is no better hit with which to compare overlap)    
    
    This iteration is terminate once we have collected N best hit, or if there are no more hit left to test for overlap 
   
   INPUT
    - tableHit         : (Panda DataFrame) Each row is a hit, with columns "TemplateName"(String),"BBox"(x,y,width,height),"Score"(float)
                        
    - scoreThreshold : Float (or None), used to remove hit with too low prediction score. 
                       If sortAscending=False (ie we use a correlation measure so we want to keep large scores) the scores above that threshold are kept
                       If True (we use a difference measure ie we want to keep low score), the scores below that threshold are kept
                       
    - N_object      : maximum number of hit to return. Default=-1, ie return all hit passing NMS 
    - maxOverlap    : float between 0 and 1, the maximal overlap authorised between 2 bounding boxes, above this value, the bounding box of lower score is deleted
    - sortAscending : use True when low score means better prediction (Difference-based score), True otherwise (Correlation score)

    OUTPUT
    Panda DataFrame with best detection after NMS, it contains max N detection (but potentially less)
    '''
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
    if N_object == float("inf"): 
        indexes  = [ index[0] for index in indexes ] # ordered by score
    else:
        indexes  = [ index[0] for index in indexes[:N_object] ]
        
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
