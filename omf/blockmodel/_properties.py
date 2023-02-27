import numpy as np
import properties


class BlockCount(properties.Array):
    def __init__(self, doc, **kw):
        super().__init__(doc, **kw, dtype=int, shape=(3,))

    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            if item < 1:
                if instance is None:
                    msg = f"block counts must be >= 1"
                else:
                    cls = instance.__class__.__name__
                    msg = f"{cls}.{self.name} counts must be >= 1"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value


class OctreeSubblockCount(BlockCount):
    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            l = np.log2(item)
            if np.trunc(l) != l:
                if instance is None:
                    msg = f"octree block counts must be powers of two"
                else:
                    cls = instance.__class__.__name__
                    msg = f"{cls}.{self.name} octree counts must be powers of two"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value


class BlockSize(properties.Array):
    def __init__(self, doc, **kw):
        super().__init__(doc, **kw, dtype=float, shape=(3,))

    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            if item <= 0.0:
                if instance is None:
                    msg = f"block size elements must be > 0.0"
                else:
                    msg = f"{instance.__class__.__name__}.{self.name} elements must be > 0.0"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value
