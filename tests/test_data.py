"""Tests for data object validation"""
import datetime

import numpy as np
import properties
import pytest

import omf


def test_scalar_array():
    """Test array init and access works correctly"""
    arr = omf.data.Array(np.array([1, 2, 3], dtype='uint8'))
    assert arr.array.dtype.kind == 'u'
    assert np.array_equal(arr.array, [1, 2, 3])
    assert arr.datatype == 'Uint8Array'
    assert arr.shape == [3]
    assert arr.size == 24


def test_boolean_array():
    """Test boolean array bits"""
    arr = omf.data.Array(np.array([[1, 1], [0, 0]], dtype='bool'))
    assert arr.array.dtype.kind == 'b'
    assert arr.datatype == 'BooleanArray'
    assert arr.shape == [2, 2]
    assert arr.size == 4


def test_datetime_list():
    arr = omf.data.StringList(['1995-08-12T18:00:00Z', '1995-08-13T18:00:00Z'])
    assert arr.datatype == 'DateTimeArray'
    assert arr.shape == [2]


def test_string_list():
    arr = omf.data.StringList(['a', 'b', 'c'])
    assert arr.datatype == 'StringArray'
    assert arr.shape == [3]
    assert arr.size == 120


def test_array_instance_prop():
    """Test ArrayInstanceProperty validates correctly"""

    class HasArray(properties.HasProperties):
        """Test class for ArrayInstanceProperty"""
        arr = omf.data.ArrayInstanceProperty(
            'Array instance',
            shape=('*', 3),
            dtype=float,
        )

    harr = HasArray()
    harr.arr = np.array([[1., 2, 3], [4, 5, 6]])
    assert harr.validate()
    assert np.array_equal(harr.arr.array, [[1., 2, 3], [4, 5, 6]])
    assert harr.arr.datatype == 'Float64Array'
    assert harr.arr.shape == [2, 3]
    assert harr.arr.size == 64*6

    with pytest.raises(properties.ValidationError):
        harr.arr = np.array([1., 2, 3])
    with pytest.raises(properties.ValidationError):
        harr.arr = np.array([[1, 2, 3], [4, 5, 6]])

def test_vector_data_dimensionality():
    vdata = omf.data.VectorData(array=[[1, 1], [2, 2], [3, 3]])
    assert vdata.array.shape == [3, 2]
    vdata = omf.data.VectorData(array=[[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert vdata.array.shape == [3, 3]
    with pytest.raises(properties.ValidationError):
        omf.data.VectorData(array=[1, 2, 3])
    with pytest.raises(properties.ValidationError):
        omf.data.VectorData(array=[[1, 2, 3, 4]])

def test_colormap():
    """Test colormap validation"""
    cmap = omf.data.Colormap()
    with pytest.raises(ValueError):
        cmap.limits = [1., 0.]
    cmap.gradient = [[0, 0, 0]]*128
    cmap.limits = [0., 1.]
    cmap.limits[0] = 2.
    with pytest.raises(ValueError):
        cmap.validate()


def test_mapped_data():
    """Test mapped data validation"""
    mdata = omf.data.MappedData()
    mdata.indices = [0, 2, 1, -1]
    mdata.legends = [
        omf.data.Legend(
            name='color',
            values=[[0, 0, 0], [1, 1, 1], [255, 255, 255]],
        ),
        omf.data.Legend(
            name='letter',
            values=['x', 'y', 'z'],
        ),
    ]
    mdata.location = 'vertices'
    assert mdata.validate()
    assert mdata.indices is mdata.array
    assert mdata.value_dict(0) == {'color': (0, 0, 0), 'letter': 'x'}
    assert mdata.value_dict(1) == {'color': (255, 255, 255), 'letter': 'z'}
    assert mdata.value_dict(2) == {'color': (1, 1, 1), 'letter': 'y'}
    assert mdata.value_dict(3) is None
    with pytest.raises(ValueError):
        mdata.array = [0.5, 1.5, 2.5]
    with pytest.raises(ValueError):
        mdata.array = [-10, 0, 1]
    mdata.array.array[0] = -10
    with pytest.raises(ValueError):
        mdata.validate()
    mdata.array.array[0] = 0
    mdata.metadata = {
        'units': 'm',
        'date_created': datetime.datetime.utcnow(),
        'version': 'v1.3',
    }
    assert mdata.validate()
    mdata.legends.append(
        omf.data.Legend(
            name='short',
            values=[0.5, 0.6],
        )
    )
    with pytest.raises(ValueError):
        mdata.validate()
