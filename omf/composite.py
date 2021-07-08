"""composite.py: Element composed of other elements"""
import properties

from .base import ProjectElement
from .blockmodel import (
    ArbitrarySubBlockModel,
    OctreeSubBlockModel,
    RegularBlockModel,
    RegularSubBlockModel,
    TensorBlockModel,
)
from .lineset import LineSetElement
from .pointset import PointSetElement
from .surface import SurfaceElement, SurfaceGridElement


class CompositeElement(ProjectElement):
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
                TensorBlockModel,
                ArbitrarySubBlockModel,
                LineSetElement,
                PointSetElement,
                SurfaceElement,
                SurfaceGridElement,
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
