"""Tests for PointSet validation"""

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_pointset_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    pts = omf.PointSetElement(
        name="Random Points",
        description="Just random points",
        geometry=omf.PointSetGeometry(vertices=np.random.rand(100, 3)),
        data=[
            omf.ScalarData(
                name="rand data", array=np.random.rand(100), location="vertices"
            ),
        ],
        color="green",
    )

    file = str(tmp_path / "pointset.geoh5")
    omf.OMFWriter(pts, file)

    with Workspace(file) as workspace:
        points = workspace.get_entity("Random Points")[0]
        np.testing.assert_array_almost_equal(
            np.r_[pts.geometry.vertices.array], points.vertices
        )

        data = points.get_entity("rand data")[0]
        np.testing.assert_array_almost_equal(np.r_[pts.data[0].array], data.values)
