from collections.abc import Iterable, Sequence
import copy
import os
import typing
from typing import Any

from numpy.typing import ArrayLike, NDArray
import numpy as np
import tifffile  # type: ignore

StrPath = os.PathLike | str

IMAGE_DIM = 2
IMAGE_HYPERSTACK_DIM = 5
VALID_DIMS = "tzcyx"


class ImageHyperStack:
    """A utility class that holds an image hyperstack."""

    _images: NDArray[np.number]
    _channels: list[str | None]
    _channel_dim: int | None

    def __init__(self, data: ArrayLike, dims: str = "tzcyx", channels: Iterable[str] | None = None):
        """Create an ImageHyperStack from an array.

        Parameters
        ----------
        data
            An array representing the image stack.
        dims
            The order in which the dimensions appear in the array. Possible dimensions are:
            channels (c, optional), time (t, optional), Z-dimension (z, optional), X-dimension (x),
            and Y-dimension (y).
            The default order is assumed to be "tzcyx" where priority is assigned to the rightmost
            dimensions if the array has less than five dimensions. So a 3-dimensional array would
            be assumed to have dimensions "cyx". dims is case-insensitive.
        channels
            List of channel names. By default channels will only be indexable by numbers.

        """
        self._images = np.array(data)
        if self._images.ndim < IMAGE_DIM or self._images.ndim > IMAGE_HYPERSTACK_DIM:
            raise ValueError(
                f"Hyperstacks must have {IMAGE_DIM} to {IMAGE_HYPERSTACK_DIM} dimensions "
                "got: {self._images.ndim} dimensions."
            )

        # Always save dimensions as upper case.
        self._dims = dims.lower()

        # Make sure we have provided enough dimensions.
        if len(self._dims) < self._images.ndim:
            raise ValueError(f"Dimensions string too short to describe data: {self._dims}.")

        # Make sure all dimensions are valid.
        if set(self._dims) - set(VALID_DIMS):
            raise ValueError(
                f"Invalid dimension names. Allowed characters: {VALID_DIMS} got: {self._dims}."
            )

        # Truncate the number of dimensions so they match the provided data.
        self._dims = self._dims[-self._images.ndim :]

        # Check that compulsory dimensions are included.
        if "y" not in self._dims or "x" not in self._dims:
            raise ValueError("Dimensions must include 'y' and 'x'")

        # Rearrange indices in order: TZCYX.
        dim_permutation = []
        for dim in ["t", "z", "c", "y", "x"]:
            if dim in self._dims:
                dim_permutation.append(self._dims.index(dim))
        self._images = np.transpose(self._images, dim_permutation)

        # Save the dimension of the channel if we have one.
        if "c" in self._dims:
            self._channel_dim = self._dims.index("c")
        else:
            self._channel_dim = None

        if self._channel_dim is not None:
            # Register the channel names.
            num_channels = self._images.shape[self._channel_dim]
            self._channels = [None] * num_channels
            if channels is not None:
                for i, channel in enumerate(channels):
                    self.rename_channel(i, channel)
                self._channels = list(channels)

    def rename_channel(self, old_name: str | int, new_name: str) -> None:
        """Rename channel name using the old name or its index.

        Parameters
        ----------
        old_name
            Original name or index of channel.
        new_name
            New name for old_name channel.

        """
        if new_name in self._channels:
            raise ValueError(f"Duplicate channel name: {new_name}.")

        # Convert the old name to an index if necessary.
        if isinstance(old_name, str):
            idx = self.channel_index(old_name)
        else:
            idx = old_name

        try:
            # Rename the channel.
            self._channels[idx] = new_name
        except IndexError:
            raise IndexError("Channel index out of range.")

    def __getitem__(self, idxs: Any) -> NDArray[np.number] | np.number:
        new_idxs = idxs
        # Check to see if we need to handle channel string indices.
        if self._channel_dim is not None:
            # Define a function that recursively changes channels to indices.
            def channels_to_idx(l: Any) -> Any:
                if isinstance(l, str):
                    return self.channel_index(l)
                elif isinstance(l, range):
                    if isinstance(l.start, str):
                        start = self.channel_index(l.start)
                    else:
                        start = l.start

                    if isinstance(l.stop, str):
                        stop = self.channel_index(l.stop)
                    else:
                        stop = l.stop

                    return range(start, stop, l.step)
                elif isinstance(l, Sequence):
                    result = []
                    for item in l:
                        result.append(channels_to_idx(item))
                    return np.array(result)
                else:
                    return l

            if isinstance(idxs, tuple) and self._channel_dim < len(idxs):
                new_idxs = (
                    idxs[: self._channel_dim]
                    + (channels_to_idx(idxs[self._channel_dim]),)
                    + idxs[self._channel_dim + 1 :]
                )
            elif self._channel_dim == 0:
                new_idxs = channels_to_idx(idxs)

        try:
            return typing.cast(NDArray[np.number] | np.number, self._images[new_idxs])
        except IndexError as e:
            if "arrays are valid indices" in str(e):
                raise IndexError(
                    "String indices are only valid in the channel dimension, otherwise " + str(e)
                )
            else:
                raise e

    def channel_index(self, channel_name: str) -> int:
        if channel_name not in self._channels:
            raise ValueError(f"Channel name does not exist: {channel_name}.")
        return self._channels.index(channel_name)

    @property
    def channels(self) -> list[str | None]:
        return self._channels.copy()

    @property
    def dims(self) -> str:
        return self._dims


def load(file_path: StrPath) -> ImageHyperStack:
    """Load an image hyperstack from an OME-TIFF file generated by micromanager 2.0.

    Parameters
    ----------
    file_path
        Path to image file to load.

    Returns
    -------
    images
        The image hyperstacks.

    """
    # Load in the tiff file.
    with tifffile.TiffFile(file_path) as f:
        if len(f.series) > 1:
            raise tifffile.TiffFileError(
                f"Files with multiple series are not supported: {file_path}."
            )
        if f.micromanager_metadata is None or "Summary" not in f.micromanager_metadata:
            raise tifffile.TiffFileError(
                f"Different format from micromanager 2.0 OME-TIFF: {file_path}."
            )

        # We have a valid micromanager-generated file, read it into memory.
        channels = f.micromanager_metadata["Summary"].get("ChNames")
        return ImageHyperStack(f.series[0].asarray(), f.series[0].axes, channels)
