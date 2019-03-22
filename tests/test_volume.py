"""Tests for Volume validation"""
import pytest

import omf


def test_volumegrid():
    """Test volume grid geometry validation"""
    geom = omf.volume.VolumeGridGeometry()
    geom.tensor_u = [1., 1.]
    geom.tensor_v = [2., 2., 2.]
    geom.tensor_w = [3.]
    assert geom.validate()
    assert geom.location_length('vertices') == 24
    assert geom.location_length('cells') == 6
    geom.axis_v = [1., 1., 0]
    with pytest.raises(ValueError):
        geom.validate()
    geom.axis_v = 'Y'
