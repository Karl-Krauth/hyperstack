import hyperstack as hs
import numpy as np
import numpy.testing as nptest


def test_defaults():
    images = hs.Hyperstack(np.ones([50, 20, 10]))
    assert isinstance(images, hs.Hyperstack)
    assert isinstance(images[0], hs.Hyperstack)
    assert images.dims == "cyx"
    assert len(images.channels) == 50
    assert images.ndim == 3
    assert images.shape == (50, 20, 10)


def test_indexing():
    arr = np.arange(1800).reshape(18, 5, 20)
    images = hs.Hyperstack(arr)

    # No indexing.
    nptest.assert_array_equal(arr, images)

    # Basic indexing.
    nptest.assert_array_equal(arr[0], images[0])
    nptest.assert_array_equal(arr[-2], images[-2])
    nptest.assert_array_equal(arr[0][2], images[0][2])
    nptest.assert_array_equal(arr[1, 3], images[1, 3])

    # Slicing and striding.
    nptest.assert_array_equal(arr[1:7:2], images[1:7:2])
    nptest.assert_array_equal(arr[-2:10], images[-2:10])
    nptest.assert_array_equal(arr[-3:3:-1], images[-3:3:-1])
    nptest.assert_array_equal(arr[5:], images[5:])

    # Dimensional indexing tools.
    nptest.assert_array_equal(arr[..., 0], images[..., 0])
    nptest.assert_array_equal(arr[:, np.newaxis, :, :], images[:, np.newaxis, :, :])

    # Integer array indexing.
    nptest.assert_array_equal(arr[np.array([3, 3, 1])], images[np.array([3, 3, 1])])
    nptest.assert_array_equal(arr[[3, 3, -3]], images[[3, 3, -3]])
    nptest.assert_array_equal(arr[[0, 2, 4], [0, 1, 2]], images[[0, 2, 4], [0, 1, 2]])
    nptest.assert_array_equal(arr[[0, 2, 4], 1], images[[0, 2, 4], 1])

    # Boolean array indexing.
    nptest.assert_array_equal(arr[arr < 5], images[images < 5])
    nptest.assert_array_equal(arr[(arr > 20)[:, 3, 1]], images[(images > 20)[:, 3, 1]])

    # Combining basic and advanced indexing.
    nptest.assert_array_equal(arr[[0, 2, 4], 1:3], images[[0, 2, 4], 1:3])
