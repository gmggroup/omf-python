"""Tests for PointSet validation"""
import numpy as np

import omf


def test_pointset():
    """Test pointset geometry validation"""
    elem = omf.pointset.PointSetElement()
    elem.vertices = np.random.rand(10, 3)
    assert elem.validate()
    assert elem.location_length('vertices') == 10
