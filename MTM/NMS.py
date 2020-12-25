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
hitScore = lambda hit: hit[0] # return the score from a hit [score, bbox, index]

def NMS(listHit, maxOverlap=0.5, nObjects=float("inf"), sortDescending=True):
    """
    Overlap-based Non-Maxima Supression for bounding-boxes.
    
    it compares the hits after maxima/minima detection, and removes the ones that are too close (too large overlap)
    This function works both with a threshold on the score, and an optional expected number of detected bbox
    
    if a scoreThreshold is specified, we first discard any hit below/above the threshold (depending on sortDescending)
    if sortDescending = True,  the hits with score above the treshold are kept (ie when high score means better prediction ex : Correlation)
    if sortDescending = False, the hits with score below the threshold are kept (ie when low score means better prediction ex : Distance measure)

    Then the hit are ordered so that we have the best hits first.
    Then we iterate over the list of hits, taking one hit at a time and checking for overlap with the previous validated hit (the Final Hit list is directly iniitialised with the first best hit as there is no better hit with which to compare overlap)    
    
    This iteration is terminate once we have collected N best hit, or if there are no more hit left to test for overlap 
    
    Parameters
    ----------
    listHit : list of lists or tuples
        list containing hits each encoded as a list [score, bbox, index], bboxes are encoded as (x,y,width, height).
    sortDescending : boolean, optional
        Should be True when high score means better prediction (Correlation score), False otherwise (Difference-based score). The default is True.
    nObjects : integer or float("inf"), optional
        Maximum number of hits to return (for instance when the number of object in the image is known)
        The default is float("inf").
    maxOverlap : float, optional
        Float between 0 and 1. 
        Maximal overlap authorised between 2 bounding boxes. Above this value, the bounding box of lower score is deleted. 
        The default is 0.5.

    Returns
    -------
    list
    List of best detections after NMS, it contains max nObjects detections (but potentially less)
    """
    if len(listHit)<=1:
        # 0 or 1 single hit passed to the function
        return listHit
    
    # Sort score to have best predictions first (ie lower score if difference-based, higher score if correlation-based)
    # important as we loop testing the best boxes against the other boxes)
    listHit.sort(reverse=sortDescending, key=hitScore)
    listHit_final  = listHit[0:1] # initialize the final list with best hit of the pool
    listHit_test   = listHit[1:]  # rest of hit to test for NMS
    
    # Loop to compute overlap
    for hitTest in listHit_test: 
        
        # stop if we collected nObjects
        if len(listHit_final) == nObjects: break

        # Get bbox of test hit
        test_bbox = hitTest[1] # a hit is [score, bbox, index]
        
        # Loop over confirmed hits to compute successively overlap with testHit
        for hitFinal in listHit_final: 
            
            # Recover Bbox from hit
            bbox2 = hitFinal[1]
            
            # Compute the Intersection over Union between test_peak and current peak
            IoU = computeIoU(test_bbox, bbox2)
            
            # Initialise the boolean value to true before test of overlap
            keepHit = True 
    
            if IoU>maxOverlap:
                keepHit = False
                #print("IoU above threshold\n")
                break # no need to test overlap with the other peaks
            
            else:
                #print("IoU below threshold\n")
                # no overlap for this particular (test_peak,peak) pair, keep looping to test the other (test_peak,peak)
                continue
        
        # After testing against all peaks (for loop is over)
        # 1) remove the hit from the hit pool
        # 2) append or not the peak to final
        #listHit_test.remove(hitTest)
        if keepHit: listHit_final.append(hitTest)

    return listHit_final

            
if __name__ == "__main__":
    listHit =[ 
            [0.8, (780,  350, 700, 480),0],
            [0.6, (806,  416, 716, 442),0],
            [0.4, (1074, 530, 680, 390),1]
            ]

    finalHits = NMS( listHit, scoreThreshold=0.3, sortDescending=True, maxOverlap=0.8, nObjects=2)
    print(finalHits)
