"""Tests for PointSet validation"""

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_surface_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    surf = omf.SurfaceElement(
        name="trisurf",
        geometry=omf.SurfaceGeometry(
            vertices=np.random.rand(100, 3),
            triangles=np.floor(np.random.rand(50, 3) * 100).astype(int),
        ),
        data=[
            omf.ScalarData(
                name="rand vert data", array=np.random.rand(100), location="vertices"
            ),
            omf.ScalarData(
                name="rand face data", array=np.random.rand(50), location="faces"
            ),
        ],
        color=[100, 200, 200],
    )
    file = str(tmp_path / "surface.geoh5")
    omf.OMFWriter(surf, file)

    with Workspace(file) as workspace:
        curve = workspace.get_entity("trisurf")[0]
        np.testing.assert_array_almost_equal(
            np.r_[surf.geometry.vertices.array], curve.vertices
        )

        data = curve.get_entity("rand vert data")[0]
        np.testing.assert_array_almost_equal(np.r_[surf.data[0].array], data.values)

        data = curve.get_entity("rand face data")[0]
        np.testing.assert_array_almost_equal(np.r_[surf.data[1].array], data.values)
