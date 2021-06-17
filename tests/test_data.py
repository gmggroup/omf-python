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
    """Test string list gives datetime datatype"""
    arr = omf.data.StringList(['1995-08-12T18:00:00Z', '1995-08-13T18:00:00Z'])
    assert arr.datatype == 'DateTimeArray'
    assert arr.shape == [2]


def test_string_list():
    """Test string list gives string datatype"""
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
    """Test only 2D and 3D arrays are valid for vector data"""
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
    with pytest.raises(properties.ValidationError):
        cmap.limits = [1., 0.]
    cmap.gradient = [[0, 0, 0]]*100
    cmap.limits = [0., 1.]
    cmap.limits[0] = 2.
    with pytest.raises(properties.ValidationError):
        cmap.validate()
    cmap.limits[0] = 0.
    cmap.validate()
    with pytest.raises(properties.ValidationError):
        cmap.gradient = np.array([[0, 0, -1]])
    with pytest.raises(properties.ValidationError):
        cmap.gradient = np.array([[0, 0, 256]])


def test_mapped_data():
    """Test mapped data validation"""
    mdata = omf.data.CategoryData()
    mdata.indices = [0, 2, 1, -1]
    mdata.categories = omf.data.Legend(
        name='letter',
        values=['x', 'y', 'z'],
    )
    mdata.location = 'vertices'
    assert mdata.validate()
    assert mdata.indices is mdata.array
    with pytest.raises(properties.ValidationError):
        mdata.array = [0.5, 1.5, 2.5]
    mdata.array = [-10, 0, 1]
    assert mdata.validate()
    mdata.array.array[0] = 0
    mdata.categories.colors = ['red', 'blue', 'green']
    mdata.metadata = {
        'units': 'm',
        'date_created': datetime.datetime.utcnow(),
        'version': 'v1.3',
    }
    assert mdata.validate()
    mdata.categories.colors = ['red', 'blue']
    with pytest.raises(properties.ValidationError):
        mdata.validate()
    with pytest.raises(properties.ValidationError):
        mdata.categories = omf.data.Legend(
            name='numeric',
            values=[0.5, 0.6, 0.7],
        )
