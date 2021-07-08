"""pointset.py: PointSet element and geometry"""
import properties

from .base import ProjectElement
from .attribute import ArrayInstanceProperty
from .texture import HasTexturesMixin


class PointSetElement(ProjectElement, HasTexturesMixin):
    """Contains point set spatial information and attributes"""

    schema = "org.omf.v2.element.pointset"

    vertices = ArrayInstanceProperty(
        "Spatial coordinates of points relative to point set origin",
        shape=("*", 3),
        dtype=float,
    )
    subtype = properties.StringChoice(
        "Category of PointSet",
        choices=("point", "collar", "blasthole"),
        default="point",
    )

    _valid_locations = ("vertices",)

    def location_length(self, location):

        """Return correct attribute length based on location"""
        return self.num_nodes

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return len(self.vertices.array)

    @property
    def num_cells(self):
        """Number of cell centers (same as nodes)"""
        return self.num_nodes
