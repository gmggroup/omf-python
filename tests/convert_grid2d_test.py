"""Tests for PointSet validation"""

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_grid2d_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    grid = omf.SurfaceElement(
        name="gridsurf",
        geometry=omf.SurfaceGridGeometry(
            tensor_u=np.ones(10).astype(float),
            tensor_v=np.ones(15).astype(float),
            origin=[50.0, 50.0, 50.0],
            axis_u=[0.73, 0, 0.73],
            axis_v=[0.73, 0.73, 0],
            offset_w=np.random.rand(11, 16).flatten(),
        ),
        data=[
            omf.ScalarData(
                name="rand vert data",
                array=np.random.rand(11, 16).flatten(),
                location="vertices",
            ),
            omf.ScalarData(
                name="rand face data",
                array=np.random.rand(10, 15).flatten(order="f"),
                location="faces",
            ),
        ],
    )
    file = str(tmp_path / "grid2d.geoh5")
    omf.OMFWriter(grid, file)

    with Workspace(file) as workspace:
        grid2d = workspace.get_entity("gridsurf")[0]

        data = grid2d.get_entity("rand vert data")[0]
        np.testing.assert_array_almost_equal(np.r_[grid.data[0].array], data.values)

        data = grid2d.get_entity("rand face data")[0]
        np.testing.assert_array_almost_equal(np.r_[grid.data[1].array], data.values)
