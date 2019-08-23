"""Tests for PointSet validation"""
import datetime
import numpy as np

import omf


def test_pointset():
    """Test pointset geometry validation"""
    elem = omf.pointset.PointSetElement()
    elem.vertices = np.random.rand(10, 3)
    assert elem.validate()
    assert elem.location_length('vertices') == 10
    elem.metadata = {
        'color': 'green',
        'date_created': str(datetime.datetime.utcnow()),
        'version': 'v1.3',
    }
    assert elem.validate()
