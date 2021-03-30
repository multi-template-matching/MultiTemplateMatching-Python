"""
Generic Detection object with functions to compute intersection, union and IoU between detections.
This class should be generic thanks to getter functions and not specific to a specific implementation
Basically any Detection that fullfills the methods below should work
Good for skimage not for opencv NMSBoxes
"""
from shapely.geometry import polygon, box

class Detection(polygon.Polygon):
    """
    Parameters
    ----------
    x,y int:
        x,y of the top left corner

    width, height: int
        dimensions of the rectangle outlining the detection

    score, float:
        detection score

    template_index, int (optional)
        positional index of the template in the iniial list of templates

    label, string (optional)
        label for the detection (e.g. a category name or template name)
    """
    def __init__(self, x, y, width, height, score, template_index=0, label=""):
        super().__init__( [(x,y), (x+width-1,y), (x+width-1, y+height-1), (x, y+height-1)] )
        self.score = score
        self.template_index = template_index
        self.label = label

    def get_label(self):
        return self.label

    def get_score(self):
        return self.score

    def get_index(self):
        return self.template_index

    def __str__(self):
        return ("({}, {}, xyxy:{})").format(self.get_label(),
                                        self.get_score(),
                                        self.bounds)

    def __repr__(self):
        return self.__str__()

    def intersection_area(self, detection2):
        return self.intersection(detection2).area

    def union_area(self, detection2):
        return self.union(detection2).area

    def intersection_over_union(self, detection2):
        return self.intersection_area(detection2)/self.union_area(detection2)

    """
    # If you were implementing a different type of Detection object
    # These 2 functions should also be implemented
    # Here they are inherited from shapely's Polygon
    def overlaps(self, detection2):
    def contains(self, detection2)
    """
if __name__ == "__main__":
    detection = Detection(0,0,10,10,0.5,label="Test")
    print(detection)
    print([detection,detection])