"""Tests for Surface validation"""
import numpy as np
import pytest

import omf


def test_surface():
    """Test surface geometry validation"""
    geom = omf.surface.SurfaceGeometry()
    geom.vertices = np.random.rand(10, 3)
    geom.triangles = np.random.randint(9, size=[5, 3])
    assert geom.validate()
    assert geom.location_length('vertices') == 10
    assert geom.location_length('faces') == 5
    geom.triangles.array[0, 0] = -1
    with pytest.raises(ValueError):
        geom.validate()
    geom.triangles.array[0, 0] = 10
    with pytest.raises(ValueError):
        geom.validate()


def test_surfacegrid():
    """Test surface grid geometry validation"""
    geom = omf.surface.SurfaceGridGeometry()
    geom.tensor_u = [1., 1.]
    geom.tensor_v = [2., 2., 2.]
    assert geom.validate()
    assert geom.location_length('vertices') == 12
    assert geom.location_length('faces') == 6
    geom.axis_v = [1., 1., 0]
    with pytest.raises(ValueError):
        geom.validate()
    geom.axis_v = 'Y'
    geom.offset_w = np.random.rand(12)
    geom.validate()
    geom.offset_w = np.random.rand(6)
    with pytest.raises(ValueError):
        geom.validate()
