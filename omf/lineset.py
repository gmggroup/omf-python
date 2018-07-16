"""lineset.py: LineSet element and geometry"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties

from .base import ProjectElement, ProjectElementGeometry
from .data import Int2Array, Vector3Array


class LineSetGeometry(ProjectElementGeometry):
    """Contains spatial information of a line set"""
    vertices = properties.Instance(
        'Spatial coordinates of line vertices relative to line set origin',
        Vector3Array
    )
    segments = properties.Instance(
        'Endpoint vertex indices of line segments',
        Int2Array
    )

    _valid_locations = ('vertices', 'segments')

    def location_length(self, location):
        """Return correct data length based on location"""
        if location == 'segments':
            return self.num_cells
        return self.num_nodes

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return len(self.vertices)

    @property
    def num_cells(self):
        """Number of cells (segments)"""
        return len(self.segments)

    @properties.validator
    def _validate_mesh(self):
        """Ensures segment indices are valid"""
        if np.min(self.segments) < 0:
            raise ValueError('Segments may only have positive integers')
        if np.max(self.segments) >= len(self.vertices):
            raise ValueError('Segments expects more vertices than provided')
        return True


class LineSetElement(ProjectElement):
    """Contains mesh, data, and options of a line set"""
    geometry = properties.Instance(
        'Structure of the line element',
        instance_class=LineSetGeometry
    )
    subtype = properties.StringChoice(
        'Category of LineSet',
        choices=('line', 'borehole'),
        default='line'
    )

    def toVTK(self):
        """Convert the line set to a ``vtkPloyData`` data object."""
        import vtk

        output = vtk.vtkPolyData()
        cells = vtk.vtkCellArray()
        pts = vtk.vtkPoints()

        for v in self.geometry.vertices:
            pts.InsertNextPoint(v[0],v[1],v[2])

        for i in range(len(self.geometry.segments)-1):
            seg = self.geometry.segments[i]
            aLine = vtk.vtkLine()
            aLine.GetPointIds().SetId(0, seg[0])
            aLine.GetPointIds().SetId(1, seg[1])
            cells.InsertNextCell(aLine)

        output.SetPoints(pts)
        output.SetLines(cells)

        # TODO: Add cell data to allow coloring by line:
        return output
