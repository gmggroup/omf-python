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
    """Placeholder Composite class that will be replaced in a few lines of code

    The properties library does not allow updating forward references
    so we need to add Composite to the valid element types after the
    class is created, then create an identical subclass so the docs
    update.

    We should switch from properties to pydantic, which does allow
    updating forward refs...
    """

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
        """Composite attributes may only be defined on each element

        Each element within the composite may also have its own
        attributes.
        """
        return len(self.elements)


composite_props = Composite._props["elements"].prop.props  # pylint: disable=E1101
Composite._props["elements"].prop.props = composite_props + (  # pylint: disable=E1101
    Composite,
)


class Composite(Composite):  # pylint: disable=E0102
    """Object constructed from other primitive elements and composites"""
