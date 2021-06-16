"""composite.py: Element composed of other elements"""
import properties

from .base import ProjectElement
from .blockmodel import RegularBlockModel
from .lineset import LineSetElement
from .pointset import PointSetElement
from .surface import SurfaceElement, SurfaceGridElement
from .volume import VolumeGridElement


class CompositeElement(ProjectElement):
    """Element constructed from other primitive elements"""
    schema_type = 'org.omf.v2.composite'

    elements = properties.List(
        'Elements grouped into one composite element',
        prop=properties.Union('', (
            RegularBlockModel,
            LineSetElement,
            PointSetElement,
            SurfaceElement,
            SurfaceGridElement,
            VolumeGridElement,
        )),
        default=list,
    )

    _valid_locations = ('elements',)

    def location_length(self, location):
        """Composite element data may only be defined on each element

        Each element within the composite element may also have its own
        data sets.
        """
        return len(self.elements)
