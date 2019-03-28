"""Tests for PointSet validation"""
import numpy as np

import omf


def test_pointset():
    """Test pointset geometry validation"""
    geom = omf.pointset.PointSetGeometry()
    geom.vertices = np.random.rand(10, 3)
    assert geom.validate()
    assert geom.location_length('vertices') == 10
