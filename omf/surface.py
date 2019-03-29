"""surface.py: Surface element and geometry"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties

from .base import ProjectElement
from .data import Int3Array, ScalarArray, Vector3Array
from .texture import ImageTexture


class BaseSurfaceElement(ProjectElement):
    """Base class for surface elements"""
    textures = properties.List(
        'Images mapped on the surface element',
        prop=ImageTexture,
        required=False,
        default=list,
    )
    subtype = properties.StringChoice(
        'Category of Surface',
        choices=('surface',),
        default='surface',
    )

    _valid_locations = ('vertices', 'faces')

    def location_length(self, location):
        """Return correct data length based on location"""
        if location == 'faces':
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


class SurfaceElement(BaseSurfaceElement):
    """Contains triangulated surface spatial information and attributes"""
    vertices = properties.Instance(
        'Spatial coordinates of vertices relative to surface origin',
        Vector3Array,
    )
    triangles = properties.Instance(
        'Vertex indices of surface triangles',
        Int3Array,
    )

    @property
    def num_nodes(self):
        """get number of nodes"""
        return len(self.vertices)

    @property
    def num_cells(self):
        """get number of cells"""
        return len(self.triangles)

    @properties.validator
    def _validate_mesh(self):
        if np.min(self.triangles.array) < 0:
            raise ValueError('Triangles may only have positive integers')
        if np.max(self.triangles.array) >= len(self.vertices.array):
            raise ValueError('Triangles expects more vertices than provided')
        return True


class SurfaceGridElement(BaseSurfaceElement):
    """Contains 2D grid spatial information and attributes"""
    tensor_u = properties.Array(
        'Grid cell widths, u-direction',
        shape=('*',),
        dtype=float,
    )
    tensor_v = properties.Array(
        'Grid cell widths, v-direction',
        shape=('*',),
        dtype=float,
    )
    axis_u = properties.Vector3(
        'Vector orientation of u-direction',
        default='X',
        length=1,
    )
    axis_v = properties.Vector3(
        'Vector orientation of v-direction',
        default='Y',
        length=1,
    )
    offset_w = properties.Instance(
        'Node offset',
        ScalarArray,
        required=False,
    )
    origin = properties.Vector3(
        'Origin of the Mesh relative to Project coordinate reference system',
        default=[0., 0., 0.],
    )

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return (len(self.tensor_u)+1) * (len(self.tensor_v)+1)

    @property
    def num_cells(self):
        """Number of cells (faces)"""
        return len(self.tensor_u) * len(self.tensor_v)

    @properties.validator
    def _validate_mesh(self):
        """Check if mesh content is built correctly"""
        if not np.abs(self.axis_u.dot(self.axis_v)) < 1e-6:                    #pylint: disable=no-member
            raise ValueError('axis_u and axis_v must be orthogonal')
        if self.offset_w is properties.undefined or self.offset_w is None:
            return True
        if len(self.offset_w.array) != self.num_nodes:
            raise ValueError(
                'Length of offset_w, {zlen}, must equal number of nodes, '
                '{nnode}'.format(
                    zlen=len(self.offset_w),
                    nnode=self.num_nodes
                )
            )
        return True
