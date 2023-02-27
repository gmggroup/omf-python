import numpy as np
import properties

from ..base import ProjectElement
from ._properties import BlockCount, BlockSize


class BaseBlockModel(ProjectElement):
    """Basic orientation properties and indexing for all block models"""

    axis_u = properties.Vector3(
        "Vector orientation of u-direction",
        default="X",
        length=1,
    )
    axis_v = properties.Vector3(
        "Vector orientation of v-direction",
        default="Y",
        length=1,
    )
    axis_w = properties.Vector3(
        "Vector orientation of w-direction",
        default="Z",
        length=1,
    )
    corner = properties.Vector3(
        "Corner of the block model relative to Project coordinate reference system",
        default="zero",
    )
    _valid_locations = ("cells", "parent_blocks")

    @properties.validator
    def _validate_axes(self):
        """Check if mesh content is built correctly"""
        if not (
            np.abs(self.axis_u.dot(self.axis_v) < 1e-6)
            and np.abs(self.axis_v.dot(self.axis_w) < 1e-6)
            and np.abs(self.axis_w.dot(self.axis_u) < 1e-6)
        ):
            raise ValueError("axis_u, axis_v, and axis_w must be orthogonal")
        return True

    @property
    def parent_block_count(self):
        """Computed property for number of parent blocks

        This is required for ijk indexing. For tensor and regular block
        models, all the blocks are considered parent blocks.
        """
        raise NotImplementedError()

    def ijk_to_index(self, ijk):
        """Map IJK triples to flat indices for a singoe triple or an array, preseving shape."""
        arr = np.asarray(ijk)
        if arr.dtype.kind not in "ui":
            raise TypeError(f"'ijk' must be integer typed, found {arr.dtype}")
        match arr.shape:
            case (*output_shape, 3):
                shaped = arr.reshape(-1, 3)
            case _:
                raise ValueError(
                    "'ijk' must have 3 elements or be an array with shape (*_, 3)"
                )
        count = self.parent_block_count
        if (shaped < 0).any() or (shaped >= count).any():
            raise IndexError(f"0 <= ijk < ({count[0]}, {count[1]}, {count[2]}) failed")
        indices = np.ravel_multi_index(multi_index=shaped.T, dims=count, order="F")
        if output_shape == ():
            return indices[0]
        else:
            return indices.reshape(output_shape)

    def index_to_ijk(self, index):
        """Map flat indices to IJK triples for a singoe index or an array, preserving shape."""
        arr = np.asarray(index)
        if arr.dtype.kind not in "ui":
            raise TypeError(f"'index' must be integer typed, found {arr.dtype}")
        output_shape = arr.shape + (3,)
        shaped = arr.reshape(-1)
        count = self.parent_block_count
        if (shaped < 0).any() or (shaped >= np.prod(count)).any():
            raise IndexError(f"0 <= index < {np.prod(count)} failed")
        ijk = np.unravel_index(indices=shaped, shape=count, order="F")
        return np.c_[ijk[0], ijk[1], ijk[2]].reshape(output_shape)


class RegularBlockModel(BaseBlockModel):
    """Block model with constant spacing in each dimension."""

    schema = "org.omf.v2.elements.blockmodel.regular"

    block_count = BlockCount("Number of blocks along u, v, and w axes")
    block_size = BlockSize("Size of blocks in the u, v, and w directions")

    @property
    def num_cells(self):
        """The number of cells, which in this case are always parent blocks."""
        return np.prod(self.parent_block_count)

    @property
    def parent_block_count(self):
        """Number of parent blocks equals number of blocks"""
        return self.block_count

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        match location:
            case "cells" | "parent_blocks" | "":
                return self.num_cells
            case _:
                raise ValueError(f"unknown location type: {location!r}")
