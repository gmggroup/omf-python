import numpy as np
import properties

from .regular import BaseBlockModel


class TensorGridBlockModel(BaseBlockModel):
    """Block model with variable spacing in each dimension.

    Unlike the rest of the block models attributes here can also be on the block vertices.
    """

    schema = "org.omf.v2.element.blockmodel.tensorgrid"

    _valid_locations = ("vertices",) + BaseBlockModel._valid_locations

    tensor_u = properties.Array(
        "Tensor cell widths, u-direction",
        shape=("*",),
        dtype=float,
    )
    tensor_v = properties.Array(
        "Tensor cell widths, v-direction",
        shape=("*",),
        dtype=float,
    )
    tensor_w = properties.Array(
        "Tensor cell widths, w-direction",
        shape=("*",),
        dtype=float,
    )

    @properties.validator("tensor_u")
    @properties.validator("tensor_v")
    @properties.validator("tensor_w")
    def _validate_tensor(self, change):
        tensor = change["value"]
        if (tensor <= 0.0).any():
            raise properties.ValidationError(
                "Tensor spacings must all be greater than zero",
                prop=change["name"],
                instance=self,
                reason="invalid",
            )

    def _require_tensors(self):
        if self.tensor_u is None or self.tensor_v is None or self.tensor_w is None:
            raise ValueError("tensors haven't been set yet")

    @property
    def parent_block_count(self):
        self._require_tensors()
        return np.array(
            (len(self.tensor_u), len(self.tensor_v), len(self.tensor_w)), dtype=int
        )

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return np.prod(self.parent_block_count + 1)

    @property
    def num_cells(self):
        """Number of cells"""
        return np.prod(self.parent_block_count)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        match location:
            case "vertices":
                return self.num_nodes
            case "cells" | "parent_blocks" | "":
                return self.num_cells
            case _:
                raise ValueError(f"unknown location type: {location!r}")
