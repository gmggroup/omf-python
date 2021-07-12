"""pointset.py: PointSet element definition"""
from .base import ProjectElement
from .attribute import ArrayInstanceProperty
from .texture import HasTexturesMixin


class PointSet(ProjectElement, HasTexturesMixin):
    """Point set element defined by vertices"""

    schema = "org.omf.v2.element.pointset"

    vertices = ArrayInstanceProperty(
        "Spatial coordinates of points relative to project origin",
        shape=("*", 3),
        dtype=float,
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
