"""volume.py: Volume element and geometry"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties

from .base import ProjectElement, ProjectElementGeometry


class VolumeGridGeometry(ProjectElementGeometry):
    """Contains spatial information of a 3D grid volume."""
    tensor_u = properties.Array(
        'Tensor cell widths, u-direction',
        shape=('*',),
        dtype=float
    )
    tensor_v = properties.Array(
        'Tensor cell widths, v-direction',
        shape=('*',),
        dtype=float
    )
    tensor_w = properties.Array(
        'Tensor cell widths, w-direction',
        shape=('*',),
        dtype=float
    )
    axis_u = properties.Vector3(
        'Vector orientation of u-direction',
        default='X',
        length=1
    )
    axis_v = properties.Vector3(
        'Vector orientation of v-direction',
        default='Y',
        length=1
    )
    axis_w = properties.Vector3(
        'Vector orientation of w-direction',
        default='Z',
        length=1
    )

    _valid_locations = ('vertices', 'cells')

    def location_length(self, location):
        """Return correct data length based on location"""
        if location == 'cells':
            return self.num_cells
        return self.num_nodes

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return ((len(self.tensor_u)+1) * (len(self.tensor_v)+1) *
                (len(self.tensor_w)+1))

    @property
    def num_cells(self):
        """Number of cells"""
        return len(self.tensor_u) * len(self.tensor_v) * len(self.tensor_w)

    @property
    def shape(self):
        return ( len(self.tensor_u), len(self.tensor_v), len(self.tensor_w))

    @properties.validator
    def _validate_mesh(self):
        """Check if mesh content is built correctly"""
        if not (np.abs(self.axis_u.dot(self.axis_v) < 1e-6) and                #pylint: disable=no-member
                np.abs(self.axis_v.dot(self.axis_w) < 1e-6) and                #pylint: disable=no-member
                np.abs(self.axis_w.dot(self.axis_u) < 1e-6)):                  #pylint: disable=no-member
            raise ValueError('axis_u, axis_v, and axis_w must be orthogonal')
        return True

    def toVTK(self):
        """Convert the 3D gridded volume to a ``vtkStructuredGrid``
        (or a ``vtkRectilinearGrid`` when apprropriate) object contatining the
        2D surface.
        """
        import vtk
        from vtk.util import numpy_support as nps

        self._validate_mesh()

        ox, oy, oz = self.origin

        def checkOrientation():
            if np.allclose(self.axis_u, (1,0,0)) and np.allclose(self.axis_v, (0,1,0)) and np.allclose(self.axis_w, (0,0,1)):
                return True
            return False

        def rotationMatrix():
            # TODO: is this correct?
            return np.array([self.axis_u, self.axis_v, self.axis_w])

        # Make coordinates along each axis
        x = ox + np.cumsum(self.tensor_u)
        x = np.insert(x, 0, ox)
        y = oy + np.cumsum(self.tensor_v)
        y = np.insert(y, 0, oy)
        z = oz + np.cumsum(self.tensor_w)
        z = np.insert(z, 0, oz)

        # If axis orientations are standard then use a vtkRectilinearGrid
        if checkOrientation():
            output = vtk.vtkRectilinearGrid()
            output.SetDimensions(len(x), len(y), len(z)) # note this subtracts 1
            output.SetXCoordinates(nps.numpy_to_vtk(num_array=x))
            output.SetYCoordinates(nps.numpy_to_vtk(num_array=y))
            output.SetZCoordinates(nps.numpy_to_vtk(num_array=z))
            return output

        # Otherwise use a vtkStructuredGrid
        output = vtk.vtkStructuredGrid()
        output.SetDimensions(len(x), len(y), len(z)) # note this subtracts 1

        # Build out all nodes in the mesh
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack( (xx.flatten(), yy.flatten(), zz.flatten()) ).T

        # Rotate the points based on the axis orientations
        rmtx = rotationMatrix()
        points = points.dot(rmtx)

        # Convert points to vtk object
        pts = vtk.vtkPoints()
        for r in points:
            pts.InsertNextPoint(r[0], r[1], r[2])
        # Now build the output
        output.SetPoints(pts)

        return output


class VolumeElement(ProjectElement):
    """Contains mesh, data, and options of a volume"""
    geometry = properties.Instance(
        'Structure of the volume element',
        instance_class=VolumeGridGeometry
    )
    subtype = properties.StringChoice(
        'Category of Volume',
        choices=('volume',),
        default='volume'
    )

    def toVTK(self):
        """Convert the volume element to a VTK data object."""
        from vtk.util import numpy_support as nps
        output = self.geometry.toVTK()
        shp = self.geometry.shape
        # Add data to output
        for data in self.data:
            arr = data.array.array
            arr = np.reshape(arr, shp).flatten(order='F')
            c = nps.numpy_to_vtk(num_array=arr, deep=True)
            c.SetName(data.name)
            output.GetCellData().AddArray(c)
        return output
