# -*- coding: utf-8 -*-
"""
Non-Maxima Supression (NMS) for match template
From a pool of bounding box each predicting possible object locations with a given score, 
the NMS removes the bounding boxes overlapping with a bounding box of higher score above the maxOverlap threshold

This effectively removes redundant detections of the same object and still allow the detection of close objects (ie small possible overlap)
The "possible" allowed overlap is set by the variable maxOverlap (between 0 and 1) which is the ratio Intersection over Union (IoU) area for a given pair of overlaping BBoxes


@author: Laurent Thomas
"""
from __future__ import division, print_function # for compatibility with Py2
import pandas as pd

def Point_in_Rectangle(Point, Rectangle):
    '''Return True if a point (x,y) is contained in a Rectangle(x, y, width, height)'''
    # unpack variables
    Px, Py = Point
    Rx, Ry, w, h = Rectangle

    return (Rx <= Px) and (Px <= Rx + w -1) and (Ry <= Py) and (Py <= Ry + h -1) # simply test if x_Point is in the range of x for the rectangle 


def computeIoU(BBox1,BBox2):
    '''
    Compute the IoU (Intersection over Union) between 2 rectangular bounding boxes defined by the top left (Xtop,Ytop) and bottom right (Xbot, Ybot) pixel coordinates
    Code adapted from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    '''
    #print('BBox1 : ', BBox1)
    #print('BBox2 : ', BBox2)
    
    # Unpack input (python3 - tuple are no more supported as input in function definition - PEP3113 - Tuple can be used in as argument in a call but the function will not unpack it automatically)
    Xleft1, Ytop1, Width1, Height1 = BBox1
    Xleft2, Ytop2, Width2, Height2 = BBox2
    
    # Compute bottom coordinates
    Xright1 = Xleft1 + Width1  -1 # we remove -1 from the width since we start with 1 pixel already (the top one)
    Ybot1    = Ytop1     + Height1 -1 # idem for the height

    Xright2 = Xleft2 + Width2  -1
    Ybot2    = Ytop2     + Height2 -1

    # determine the (x, y)-coordinates of the top left and bottom right points of the intersection rectangle
    Xleft  = max(Xleft1, Xleft2)
    Ytop   = max(Ytop1, Ytop2)
    Xright = min(Xright1, Xright2)
    Ybot   = min(Ybot1, Ybot2)
    
    # Compute boolean for inclusion
    BBox1_in_BBox2 = Point_in_Rectangle((Xleft1, Ytop1), BBox2) and Point_in_Rectangle((Xleft1, Ybot1), BBox2) and Point_in_Rectangle((Xright1, Ytop1), BBox2) and Point_in_Rectangle((Xright1, Ybot1), BBox2)
    BBox2_in_BBox1 = Point_in_Rectangle((Xleft2, Ytop2), BBox1) and Point_in_Rectangle((Xleft2, Ybot2), BBox1) and Point_in_Rectangle((Xright2, Ytop2), BBox1) and Point_in_Rectangle((Xright2, Ybot2), BBox1) 
    
    # Check that for the intersection box, Xtop,Ytop is indeed on the top left of Xbot,Ybot
    if BBox1_in_BBox2 or BBox2_in_BBox1:
        #print('One BBox is included within the other')
        IoU = 1
    
    elif Xright<Xleft or Ybot<Ytop : # it means that there is no intersection (bbox is inverted)
        #print('No overlap')
        IoU = 0 
    
    else:
        # Compute area of the intersecting box
        Inter = (Xright - Xleft + 1) * (Ybot - Ytop + 1) # +1 since we are dealing with pixels. See a 1D example with 3 pixels for instance
        #print('Intersection area : ', Inter)

        # Compute area of the union as Sum of the 2 BBox area - Intersection
        Union = Width1 * Height1 + Width2 * Height2 - Inter
        #print('Union : ', Union)
        
        # Compute Intersection over union
        IoU = Inter/Union
    
    #print('IoU : ',IoU)
    return IoU


# Helper function for the sorting of the list based on score
hitScore = lambda hit: hit[1]

def NMS(listHit, scoreThreshold=None, sortDescending=True, N_object=float("inf"), maxOverlap=0.5):
    '''
    Perform Non-Maxima supression : it compares the hits after maxima/minima detection, and removes the ones that are too close (too large overlap)
    This function works both with an optionnal threshold on the score, and number of detected bbox

    if a scoreThreshold is specified, we first discard any hit below/above the threshold (depending on sortDescending)
    if sortDescending = True,  the hits with score above the treshold are kept (ie when high score means better prediction ex : Correlation)
    if sortDescending = False, the hits with score below the threshold are kept (ie when low score means better prediction ex : Distance measure)

    Then the hit are ordered so that we have the best hits first.
    Then we iterate over the list of hits, taking one hit at a time and checking for overlap with the previous validated hit (the Final Hit list is directly iniitialised with the first best hit as there is no better hit with which to compare overlap)    
    
    This iteration is terminate once we have collected N best hit, or if there are no more hit left to test for overlap 
   
   INPUT
    - tableHit         : (Panda DataFrame) Each row is a hit, with columns "TemplateName"(String),"BBox"(x,y,width,height),"Score"(float)
                        
    - scoreThreshold : Float (or None), used to remove hit with too low prediction score. 
                       If sortDescending=True (ie we use a correlation measure so we want to keep large scores) the scores above that threshold are kept
                       While if we use sortDescending=False (we use a difference measure ie we want to keep low score), the scores below that threshold are kept
                       
    - N_object                 : number of best hit to return (by increasing score). Min=1, eventhough it does not really make sense to do NMS with only 1 hit
    - maxOverlap    : float between 0 and 1, the maximal overlap authorised between 2 bounding boxes, above this value, the bounding box of lower score is deleted
    - sortAscending : use True when low score means better prediction (Difference-based score), True otherwise (Correlation score)

    OUTPUT
    Panda DataFrame with best detection after NMS, it contains max N detection (but potentially less)
    '''
    
    # Apply threshold on prediction score
    if sortDescending : # We keep hit above the threshold
        listHit = [hit for hit in listHit if hit[1]>=scoreThreshold] # hit[1] is the score
    
    else : # We keep rows above the threshold
        listHit = [hit for hit in listHit if hit[1]<=scoreThreshold]
    
    # Sort score to have best predictions first (ie lower score if difference-based, higher score if correlation-based)
    # important as we loop testing the best boxes against the other boxes)
    listHit.sort(reverse=sortDescending, key=hitScore)
    
    # Split the inital pool into Final Hit that are kept and restTable that can be tested
    # Initialisation : 1st keep is kept for sure, restTable is the rest of the list
    #print("\nInitialise final hit list with first best hit")
    # TO DO: test that the list is long enough
    hitFinal  = listHit[0:1] # initialize the final list with best hit of the pool
    hitPool   = listHit[1:]
    
    # Loop to compute overlap
    while len(hitFinal)<N_object and hitPool: # second condition is hitPool is not empty
        
        # Report state of the loop
        #print("\n\n\nNext while iteration")
        
        #print("-> Final hit list")
        #for hit in outTable: print(hit)
        
        #print("\n-> Remaining hit list")
        #for hit in restTable: print(hit)
        
        # Test next hit (always the first of the hit pool)
        hitTest   = hitPool[0]
        
        # Get bbox of test hit
        test_bbox = hitTest[2] # a hit is [index, score, bbox]
        #print("\nTest BBox:{} for overlap against higher score bboxes".format(test_bbox))
         
        # Loop over hit in outTable to compute successively overlap with testHit
        for hit in hitFinal: 
            
            # Recover Bbox from hit
            bbox2 = hit[2]
            
            # Compute the Intersection over Union between test_peak and current peak
            IoU = computeIoU(test_bbox, bbox2)
            
            # Initialise the boolean value to true before test of overlap
            ToAppend = True 
    
            if IoU>maxOverlap:
                ToAppend = False
                #print("IoU above threshold\n")
                break # no need to test overlap with the other peaks
            
            else:
                #print("IoU below threshold\n")
                # no overlap for this particular (test_peak,peak) pair, keep looping to test the other (test_peak,peak)
                continue
        
        # After testing against all peaks (for loop is over)
        # 1) remove the hit from the hit pool
        # 2) append or not the peak to final
        hitPool.remove(hitTest)
        if ToAppend: hitFinal.append(hitTest)

    return hitFinal

            
if __name__ == "__main__":
    ListHit =[ 
            [1, 0.8, (780,  350, 700, 480)],
            [1, 0.6, (806,  416, 716, 442)],
            [1, 0.4, (1074, 530, 680, 390)]
            ]

    FinalHits = NMS( ListHit, scoreThreshold=0.7, sortDescending=True, maxOverlap=0.5 )
    print(FinalHits)
