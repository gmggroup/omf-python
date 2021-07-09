"""composite.py: Element composed of other elements"""
import properties

from .base import ProjectElement
from .blockmodel import (
    ArbitrarySubBlockModel,
    OctreeSubBlockModel,
    RegularBlockModel,
    RegularSubBlockModel,
    TensorGridBlockModel,
)
from .lineset import LineSet
from .pointset import PointSet
from .surface import Surface, TensorGridSurface


class Composite(ProjectElement):
    """Element constructed from other primitive elements"""

    schema = "org.omf.v2.composite"

    elements = properties.List(
        "Elements grouped into one composite element",
        prop=properties.Union(
            "",
            (
                RegularBlockModel,
                RegularSubBlockModel,
                OctreeSubBlockModel,
                TensorGridBlockModel,
                ArbitrarySubBlockModel,
                LineSet,
                PointSet,
                Surface,
                TensorGridSurface,
            ),
        ),
        default=list,
    )

    _valid_locations = ("elements",)

    def location_length(self, location):
        """Composite element attributes may only be defined on each element

        Each element within the composite element may also have its own
        attributes.
        """
        return len(self.elements)
