"""blockmodel/index.py: functions for handling block indexes."""
import numpy as np

__all__ = ["ijk_to_index", "index_to_ijk"]


def ijk_to_index(block_count, ijk):
    """Maps IJK index to a flat index, scalar or array."""
    arr = np.asarray(ijk)
    if arr.dtype.kind not in "ui":
        raise TypeError(f"'ijk' must be integer typed, found {arr.dtype}")
    if not arr.shape or arr.shape[-1] != 3:
        raise ValueError("'ijk' must have 3 elements or be an array with shape (*_, 3)")
    output_shape = arr.shape[:-1]
    shaped = arr.reshape(-1, 3)
    if (shaped < 0).any() or (shaped >= block_count).any():
        raise IndexError(f"0 <= ijk < ({block_count[0]}, {block_count[1]}, {block_count[2]}) failed")
    indices = np.ravel_multi_index(multi_index=shaped.T, dims=block_count, order="F")
    return indices[0] if output_shape == () else indices.reshape(output_shape)


def index_to_ijk(block_count, index):
    """Maps flat index to a IJK index, scalar or array."""
    arr = np.asarray(index)
    if arr.dtype.kind not in "ui":
        raise TypeError(f"'index' must be integer typed, found {arr.dtype}")
    output_shape = arr.shape + (3,)
    shaped = arr.reshape(-1)
    if (shaped < 0).any() or (shaped >= np.prod(block_count)).any():
        raise IndexError(f"0 <= index < {np.prod(block_count)} failed")
    ijk = np.unravel_index(indices=shaped, shape=block_count, order="F")
    return np.c_[ijk[0], ijk[1], ijk[2]].reshape(output_shape)
