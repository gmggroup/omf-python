"""Tests for Volume validation"""
import pytest

import omf


def test_volumegrid():
    """Test volume grid geometry validation"""
    elem = omf.volume.VolumeGridElement()
    elem.tensor_u = [1., 1.]
    elem.tensor_v = [2., 2., 2.]
    elem.tensor_w = [3.]
    assert elem.validate()
    assert elem.location_length('vertices') == 24
    assert elem.location_length('cells') == 6
    elem.axis_v = [1., 1., 0]
    with pytest.raises(ValueError):
        elem.validate()
    elem.axis_v = 'Y'
