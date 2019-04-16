"""Tests for Surface validation"""
import numpy as np
import pytest

import omf


def test_surface():
    """Test surface geometry validation"""
    elem = omf.surface.SurfaceElement()
    elem.vertices = np.random.rand(10, 3)
    elem.triangles = np.random.randint(9, size=[5, 3])
    assert elem.validate()
    assert elem.location_length('vertices') == 10
    assert elem.location_length('faces') == 5
    elem.triangles.array[0, 0] = -1
    with pytest.raises(ValueError):
        elem.validate()
    elem.triangles.array[0, 0] = 10
    with pytest.raises(ValueError):
        elem.validate()


def test_surfacegrid():
    """Test surface grid geometry validation"""
    elem = omf.surface.SurfaceGridElement()
    elem.tensor_u = [1., 1.]
    elem.tensor_v = [2., 2., 2.]
    assert elem.validate()
    assert elem.location_length('vertices') == 12
    assert elem.location_length('faces') == 6
    elem.axis_v = [1., 1., 0]
    with pytest.raises(ValueError):
        elem.validate()
    elem.axis_v = 'Y'
    elem.offset_w = np.random.rand(12)
    elem.validate()
    elem.offset_w = np.random.rand(6)
    with pytest.raises(ValueError):
        elem.validate()
