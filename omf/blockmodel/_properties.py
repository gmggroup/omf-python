"""blockmodel/_properties.py: block model specific property classes."""
import numpy as np
import properties


class BlockCount(properties.Array):
    """Number of blocks in three axes. Must be greater than 1."""

    def __init__(self, doc, **kw):
        super().__init__(doc, **kw, dtype=int, shape=(3,))

    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            if item < 1:
                raise properties.ValidationError("block counts must be >= 1", prop=self.name, instance=instance)
        return value


class SubBlockCount(BlockCount):
    """Number of sub-blocks in three axes. Must be between 1 and 65535."""

    def validate(self, instance, value):
        value = super().validate(instance, value)
        for item in value:
            if item > 65535:
                raise properties.ValidationError(
                    "sub-block counts are limited to 65535 in each direction",
                    prop=self.name,
                    instance=instance,
                )
        return value


class OctreeSubblockCount(SubBlockCount):
    """Number of octree sub-blocks in three axes. Must be between 1 and 65535 and a power of 2."""

    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            log = np.log2(item)
            if np.trunc(log) != log:
                if instance is None:
                    msg = "octree sub-block counts must be powers of two"
                else:
                    cls = instance.__class__.__name__
                    msg = f"{cls}.{self.name} octree counts must be powers of two"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value


class BlockSize(properties.Array):
    """Block size in three axes. Must be greater than zero."""

    def __init__(self, doc, **kw):
        super().__init__(doc, **kw, dtype=float, shape=(3,))

    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            if item <= 0.0:
                if instance is None:
                    msg = "block size elements must be > 0.0"
                else:
                    msg = f"{instance.__class__.__name__}.{self.name} elements must be > 0.0"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value


class TensorArray(properties.Array):
    """Arrays of block spacings in one axis. All spacings must be greater than zero."""

    def __init__(self, doc, **kw):
        super().__init__(doc, **kw, dtype=float, shape=("*",))

    def validate(self, instance, value):
        """Check that tensor spacings are all > 0."""
        super().validate(instance, value)
        value = np.asarray(value)
        if (value <= 0.0).any():
            raise properties.ValidationError(
                "Tensor spacings must be greater than zero",
                prop=self.name,
                instance=instance,
                reason="invalid",
            )
        return value
