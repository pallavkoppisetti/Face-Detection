import numpy as np


def integral_image(img: np.ndarray) -> np.ndarray:
    """Compute the integral image
    @img: Image whose integral image is to be calculated
    """

    iimg = np.zeros_like(img, dtype=np.int64)
    rowsum = np.zeros_like(img, dtype=np.int64)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            rowsum[x, y] = img[x, y] + rowsum[x, y - 1]
            iimg[x, y] = rowsum[x, y] + iimg[x - 1, y]

    return iimg


def test_integral_image():
    """Test the integral image function."""
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]], dtype=np.int32)

    assert np.all(integral_image(img) == iimg)


if __name__ == "__main__":
    test_integral_image()
