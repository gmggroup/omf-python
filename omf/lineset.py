"""lineset.py: LineSet element definition"""
import numpy as np
import properties

from .base import ProjectElement
from .attribute import ArrayInstanceProperty


class LineSet(ProjectElement):
    """Line set element defined by vertices connected by segments"""

    schema = "org.omf.v2.element.lineset"

    vertices = ArrayInstanceProperty(
        "Spatial coordinates of line vertices relative to project origin",
        shape=("*", 3),
        dtype=float,
    )
    segments = ArrayInstanceProperty(
        "Endpoint vertex indices of line segments; if segments is not "
        "specified, the vertices are connected in order, equivalent to "
        "segments=[[0, 1], [1, 2], [2, 3], ...]",
        shape=("*", 2),
        dtype=int,
        required=False,
    )

    _valid_locations = ("vertices", "segments")

    def location_length(self, location):
        """Return correct attribute length based on location"""
        if location == "segments":
            return self.num_cells
        return self.num_nodes

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return len(self.vertices.array)

    @property
    def num_cells(self):
        """Number of cells (segments)"""
        if self.segments is None:
            return len(self.vertices.array) - 1
        return len(self.segments.array)

    @properties.validator
    def _validate_mesh(self):
        """Ensures segment indices are valid"""
        if self.segments is None:
            return True
        if np.min(self.segments.array) < 0:
            raise properties.ValidationError("Segments may only have positive integers")
        if np.max(self.segments.array) >= len(self.vertices.array):
            raise properties.ValidationError(
                "Segments expects more vertices than provided"
            )
        return True
