"""surface.py: Surface element definitions"""
import numpy as np
import properties

from .base import ProjectElement
from .attribute import ArrayInstanceProperty
from .texture import HasTexturesMixin


class BaseSurfaceElement(ProjectElement, HasTexturesMixin):
    """Base class for surface elements"""

    _valid_locations = ("vertices", "faces")

    def location_length(self, location):
        """Return correct attribute length based on location"""
        if location == "faces":
            return self.num_cells
        return self.num_nodes

    @property
    def num_nodes(self):
        """get number of nodes"""
        raise NotImplementedError()

    @property
    def num_cells(self):
        """get number of cells"""
        raise NotImplementedError()


class Surface(BaseSurfaceElement):  # pylint: disable=R0901
    """Surface element defined by vertices connected by triangles"""

    schema = "org.omf.v2.element.surface"

    vertices = ArrayInstanceProperty(
        "Spatial coordinates of vertices relative to project origin",
        shape=("*", 3),
        dtype=float,
    )
    triangles = ArrayInstanceProperty(
        "Vertex indices of surface triangles",
        shape=("*", 3),
        dtype=int,
    )

    @property
    def num_nodes(self):
        """get number of nodes"""
        return len(self.vertices.array)

    @property
    def num_cells(self):
        """get number of cells"""
        return len(self.triangles.array)

    @properties.validator
    def _validate_mesh(self):
        """Ensure triangles values are valid indices"""
        if np.min(self.triangles.array) < 0:
            raise properties.ValidationError(
                "Triangles may only have positive integers"
            )
        if np.max(self.triangles.array) >= len(self.vertices.array):
            raise properties.ValidationError(
                "Triangles expects more vertices than provided"
            )
        return True


class TensorGridSurface(BaseSurfaceElement):  # pylint: disable=R0901
    """Surface element defined by grid with variable spacing in both dimensions"""

    schema = "org.omf.v2.element.surfacetensorgrid"

    tensor_u = properties.List(
        "Grid cell widths, u-direction",
        properties.Float("", min=0.0),
        coerce=True,
    )
    tensor_v = properties.List(
        "Grid cell widths, v-direction",
        properties.Float("", min=0.0),
        coerce=True,
    )
    axis_u = properties.Vector3(
        "Vector orientation of u-direction",
        default="X",
        length=1,
    )
    axis_v = properties.Vector3(
        "Vector orientation of v-direction",
        default="Y",
        length=1,
    )
    offset_w = ArrayInstanceProperty(
        "Node offset",
        shape=("*",),
        dtype=float,
        required=False,
    )
    corner = properties.Vector3(
        "Corner (origin) of the Mesh relative to Project coordinate reference system",
        default=[0.0, 0.0, 0.0],
    )

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return (len(self.tensor_u) + 1) * (len(self.tensor_v) + 1)

    @property
    def num_cells(self):
        """Number of cells (faces)"""
        return len(self.tensor_u) * len(self.tensor_v)

    @properties.validator
    def _validate_mesh(self):
        """Check if mesh content is built correctly"""
        if not np.abs(self.axis_u.dot(self.axis_v)) < 1e-6:
            raise properties.ValidationError("axis_u and axis_v must be orthogonal")
        if self.offset_w is properties.undefined or self.offset_w is None:
            return True
        if len(self.offset_w.array) != self.num_nodes:
            raise properties.ValidationError(
                "Length of offset_w, {zlen}, must equal number of nodes, "
                "{nnode}".format(zlen=len(self.offset_w.array), nnode=self.num_nodes)
            )
        return True
