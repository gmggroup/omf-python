"""Tests for data object validation"""
import datetime
import pytest

import omf


def test_scalar_array():
    """Test array init and access works correctly"""
    arr = omf.data.ScalarArray([1, 2, 3])
    assert arr.array.dtype.kind == 'i'
    assert len(arr) == len(arr.array)
    for i in range(3):
        assert arr[i] == arr.array[i]


def test_colormap():
    """Test colormap validation"""
    cmap = omf.data.ScalarColormap()
    with pytest.raises(ValueError):
        cmap.gradient = [[0, 0, 0], [1, 1, 1]]
    with pytest.raises(ValueError):
        cmap.limits = [1., 0.]
    cmap.gradient = [[0, 0, 0]]*128
    cmap.limits = [0., 1.]
    cmap.limits[0] = 2.
    with pytest.raises(ValueError):
        cmap.validate()


def test_color_data_clip():
    """Test color data clipping"""
    cdata = omf.data.ColorData()
    cdata.array = [[1000, 1000, -100]]
    assert cdata.array.array[0, 0] == 255
    assert cdata.array.array[0, 1] == 255
    assert cdata.array.array[0, 2] == 0


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
        'date_created': str(datetime.datetime.utcnow()),
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
