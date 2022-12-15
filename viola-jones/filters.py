import numpy as np

from integral_image import integral_image

def regional_sum(iimg: np.ndarray, top_left: tuple, bottom_right: tuple) -> float:
    """
    Calculates the sum of a region given an integral image.
    @iimg: Integral image representation of image
    @top_left: tuple (x,y) representing top left corner of image
    @bottom_right: tuple (x,y) representing bottom right corner of image 
    """
    # calculate the sum of the region
    sum = iimg[bottom_right]

    if top_left[0] > 0:
        sum -= iimg[top_left[0]-1, bottom_right[1]]
    if top_left[1] > 0:
        sum -= iimg[bottom_right[0], top_left[1]-1]
    if top_left[0] > 0 and top_left[1] > 0:
        sum += iimg[top_left[0]-1, top_left[1]-1]

    return sum

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.tl = (x, y)
        self.tr = (x, y + w)
        self.bl = (x + h, y)
        self.br = (x + h, y + w)
    

class HaarFilter:
    """
    Define a filter class that can be used to calculate the sum of a region
    @width: Width of the filter
    @height: Height of the filter
    @white_rect: List of Rects representing white rectangles
    @black_rect: List of Rects representing black rectangles
    """
    def __init__(self, white_rect, black_rect):
        self.white_rect = white_rect
        self.black_rect = black_rect

    def apply(self, iimg):
        white_sum:float = 0
        black_sum:float = 0

        for rect in self.white_rect:

            top_left = (rect.x ,rect.y)
            bottom_right = (rect.br[0], rect.br[1])

            white_sum += regional_sum(iimg, top_left, bottom_right)

        for rect in self.black_rect:

            top_left = (rect.x , rect.y)
            bottom_right = (rect.br[0], rect.br[1])

            black_sum += regional_sum(iimg, top_left, bottom_right)

        return white_sum - black_sum


class TwoColumnFilter(HaarFilter):
    def __init__(self, x,y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
        if (w %2 != 0):
            raise ValueError("Width must be even")
        super().__init__([Rect(x,y,w//2-1,h - 1)], [Rect(x,y + w//2,w//2 - 1,h - 1)])

    def __str__(self) -> str:
        return f"TwoColumnFilter({self.x}, {self.y}, {self.w}, {self.h})"

class TwoRowFilter(HaarFilter):
    def __init__(self,x,y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        if (h %2 != 0):
            raise ValueError("Height must be even")
        super().__init__([Rect(x,y,w-1,h//2 - 1)], [Rect(x + h//2,y,w-1,h//2 - 1)])

    def __str__(self) -> str:
        return f"TwoRowFilter({self.x}, {self.y}, {self.w}, {self.h})"

class ThreeColumnFilter(HaarFilter):
    def __init__(self,x,y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        if w%3 != 0:
            raise ValueError("Width must be divisible by 3")

        super().__init__([Rect(x,y,w//3 - 1,h - 1), Rect(x,y + 2*w//3,w//3 - 1,h - 1)], [Rect(x, y + w//3,w//3 - 1,h - 1)])

    def __str__(self) -> str:
        return f"ThreeColumnFilter({self.x}, {self.y}, {self.w}, {self.h})"


class QuadFilter(HaarFilter):
    def __init__(self,x,y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        if w != h:
            raise ValueError("Width and height must be equal")
        if w%2 != 0:
            raise ValueError("Width and height must be even")
        super().__init__([Rect(x,y,w//2 - 1,h//2 - 1), Rect(x + h//2,y + w//2,w//2 - 1,h//2 - 1)],
                         [Rect(x,y+ w//2,w//2 - 1,h//2 - 1), Rect(x + h//2,y,w//2 - 1,h//2 - 1)])

    def __str__(self) -> str:
        return f"QuadFilter({self.x}, {self.y}, {self.w}, {self.h})"


def build_features(width, height):
    features = []

    for w in range(1, width+1):
        for h in range(1, height+1):
            for y in range(0, width - w+1):
                for x in range(0, height - h+1):
                    try:
                        features.append(TwoColumnFilter(x,y,w,h))
                    except ValueError:
                        pass

                    try:
                        features.append(TwoRowFilter(x,y,w,h))
                    except ValueError:
                        pass

                    try:
                        features.append(ThreeColumnFilter(x,y,w,h))
                    except ValueError:
                        pass

                    try:
                        features.append(QuadFilter(x,y,w,h))
                    except ValueError:
                        pass

    return features

def apply_features(iimg, filters):
    """
    iimg - integral image
    filters - list of filters
    """

    result = []

    for filter in filters:
        try:
            score = filter.apply(iimg)
        except:
            print("Error applying filter: ", filter)
        result.append(score)

    return result            

def test_apply_features():
    iimg = np.zeros((24,24))
    features = build_features(24, 24)
    result = apply_features(iimg, features)
    print("Number of results: ", len(result))

def test_build_features():
    features = build_features(24, 24)
    print("Number of features: ", len(features))


def test_regional_sum():

    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    s = regional_sum(iimg, (1,1), (2,2))
    assert(s == 28)
    print("Regional sum test passed")

def test_two_column():
    # img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = TwoColumnFilter(0,0,2, 2)
    score = filter.apply(iimg)
    assert(score == -2)
    print("Two column filter test passed")

def test_two_row():
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = TwoRowFilter(0,0,2, 2)
    score = filter.apply(iimg)

    assert(score == -6)
    print("Two row filter test passed")

def test_three_column():
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = ThreeColumnFilter(0,0,3, 3)
    score = filter.apply(iimg)

    assert(score == 15)
    print("Three column filter test passed")

def test_quad_filter():
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = QuadFilter(0,0,2, 2)
    score = filter.apply(iimg)

    assert(score == 0)
    print("Quad filter test passed")


if __name__ == "__main__":
    test_regional_sum()
    test_two_column()
    test_two_row()
    test_three_column()
    test_quad_filter()
    test_build_features()
    test_apply_features()
