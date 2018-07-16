"""pointset.py: PointSet element and geometry"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties

from .base import ProjectElement, ProjectElementGeometry
from .data import Vector3Array
from .texture import ImageTexture


class PointSetGeometry(ProjectElementGeometry):
    """Contains spatial information of a point set"""
    vertices = properties.Instance(
        'Spatial coordinates of points relative to point set origin',
        Vector3Array
    )

    _valid_locations = ('vertices',)

    def location_length(self, location):
        """Return correct data length based on location"""
        return self.num_nodes

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return len(self.vertices)

    @property
    def num_cells(self):
        """Number of cell centers (same as nodes)"""
        return self.num_nodes


class PointSetElement(ProjectElement):
    """Contains mesh, data, textures, and options of a point set"""
    geometry = properties.Instance(
        'Structure of the point set element',
        instance_class=PointSetGeometry
    )
    textures = properties.List(
        'Images mapped on the element',
        prop=ImageTexture,
        required=False,
        default=list,
    )
    subtype = properties.StringChoice(
        'Category of PointSet',
        choices=('point', 'collar', 'blasthole'),
        default='point'
    )

    def toVTK(self):
        """Convert the point set to a ``vtkPloyData`` data object
        which contatins the point set."""
        import vtk
        import numpy as np
        from vtk.util import numpy_support as nps

        points = self.geometry.vertices
        npoints = self.geometry.num_nodes

        # Make VTK cells array
        cells = np.hstack((np.ones((npoints, 1)),
                           np.arange(npoints).reshape(-1, 1)))
        cells = np.ascontiguousarray(cells, dtype=np.int64)
        vtkcells = vtk.vtkCellArray()
        vtkcells.SetCells(npoints, nps.numpy_to_vtkIdTypeArray(cells, deep=True))

        # Convert points to vtk object
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(self.geometry.num_nodes)
        pts.SetData(nps.numpy_to_vtk(points))

        # Create polydata
        output = vtk.vtkPolyData()
        output.SetPoints(pts)
        output.SetVerts(vtkcells)

        # TODO: handle textures

        # Now add point data:
        for data in self.data:
            arr = data.array.array
            c = nps.numpy_to_vtk(num_array=arr)
            c.SetName(data.name)
            output.GetPointData().AddArray(c)

        return output
