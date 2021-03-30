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


def computeIoU(detection1, detection2):
    '''
    Compute the IoU (Intersection over Union) between 2 rectangular bounding boxes defined by the top left (Xtop,Ytop) and bottom right (Xbot, Ybot) pixel coordinates
    Code adapted from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    '''
    if not (detection1.overlaps(detection2) or
            detection1.contains(detection2) or
            detection2.contains(detection1)):
        return 0

    return detection1.intersection_over_union(detection2)


# Helper function for the sorting of the list based on score
getScore = lambda detection: detection.get_score()

def NMS(listDetections, maxOverlap=0.5, nObjects=float("inf"), sortDescending=True):
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
    listDetections : list of lists or tuples
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
    if len(listDetections)<=1:
        # 0 or 1 single hit passed to the function
        return listDetections

    # Sort score to have best predictions first (ie lower score if difference-based, higher score if correlation-based)
    # important as we loop testing the best boxes against the other boxes)
    listDetections.sort(reverse=sortDescending, key=getScore)
    listDetections_final  = listDetections[0:1] # initialize the final list with best hit of the pool
    listDetections_test   = listDetections[1:]  # rest of hit to test for NMS

    # Loop to compute overlap
    for testDetection in listDetections_test:

        # stop if we collected nObjects
        if len(listDetections_final) == nObjects:
            break

        # Loop over confirmed hits to compute successively overlap with testHit
        for finalDetection in listDetections_final:

            # Compute the Intersection over Union between test_detection and final_detection
            IoU = computeIoU(testDetection, finalDetection)

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

        # Keep detection if tested against all final detections (for loop is over)
        if keepHit:
            listDetections_final.append(testDetection)

    return listDetections_final


if __name__ == "__main__":
    from Detection import Detection
    listDetections = [
        Detection(780, 350, 700, 480, 0.8),
        Detection(806, 416, 716, 442, 0.6),
        Detection(1074, 530, 680, 390, 0.4)
        ]

    finalHits = NMS(listDetections,
                    sortDescending=True,
                    maxOverlap=0.5,
                    nObjects=2)
    print(finalHits)
