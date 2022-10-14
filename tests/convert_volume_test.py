"""Tests for PointSet validation"""

import numpy as np
from geoh5py.workspace import Workspace

import omf


def test_volume_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    vol = omf.VolumeElement(
        name="vol",
        geometry=omf.VolumeGridGeometry(
            tensor_u=np.ones(10).astype(float),
            tensor_v=np.ones(15).astype(float),
            tensor_w=np.ones(20).astype(float),
            origin=[10.0, 10.0, -10],
        ),
        data=[
            omf.ScalarData(
                name="Random Data",
                location="cells",
                array=np.arange(10 * 15 * 20).flatten(),
            )
        ],
    )

    file = str(tmp_path / "block_model.geoh5")
    omf.OMFWriter(vol, file)

    with Workspace(file) as workspace:
        block_model = workspace.get_entity("vol")[0]
        data = block_model.get_entity("Random Data")[0]
        np.testing.assert_array_almost_equal(np.r_[vol.data[0].array], data.values)
