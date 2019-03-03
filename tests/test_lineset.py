"""Tests for LineSet validation"""
import numpy as np
import pytest

import omf


def test_lineset():
    """Test lineset geometry validation"""
    geom = omf.lineset.LineSetGeometry()
    geom.vertices = np.random.rand(10, 3)
    geom.segments = np.random.randint(9, size=[5, 2])
    assert geom.validate()
    assert geom.location_length('vertices') == 10
    assert geom.location_length('segments') == 5
    geom.segments.array[0, 0] = -1
    with pytest.raises(ValueError):
        geom.validate()
    geom.segments.array[0, 0] = 10
    with pytest.raises(ValueError):
        geom.validate()
