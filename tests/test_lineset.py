"""Tests for LineSet validation"""
import numpy as np
import pytest

import omf


def test_lineset():
    """Test lineset geometry validation"""
    elem = omf.lineset.LineSetElement()
    elem.vertices = np.random.rand(10, 3)
    elem.segments = np.random.randint(9, size=[5, 2])
    assert elem.validate()
    assert elem.location_length('vertices') == 10
    assert elem.location_length('segments') == 5
    elem.segments.array[0, 0] = -1
    with pytest.raises(ValueError):
        elem.validate()
    elem.segments.array[0, 0] = 10
    with pytest.raises(ValueError):
        elem.validate()
