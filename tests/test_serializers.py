"""Tests for file serializers"""
import io
try:
    from unittest import mock
except ImportError:
    import mock

import numpy as np
import pytest

import omf


@pytest.mark.parametrize(('arr', 'mode'), [
    ([1, 2, 3], 'wb'),
    (np.array(['a', 'b', 'c']), 'wb'),
    (np.array([1, 2, 3]), 'w'),
])
def test_bad_array_serializer(arr, mode):
    """Test array serializer failures"""
    open_file = mock.MagicMock(
        mode=mode,
        write=lambda _: None,
        tell=lambda: 0,
    )
    with pytest.raises(ValueError):
        omf.serializers.array_serializer(arr, open_file)


@pytest.mark.parametrize(('arr', 'dtype'), [
    (np.array([1., 2., 3.]), '<f8'),
    (np.array([np.nan, 2., 3.]), '<f8'),
    (np.array([1, 2, 3]), '<i8'),
])
def test_array_serializer(arr, dtype):
    """Test array serializer creates correct output index"""
    open_file = mock.MagicMock(
        mode='wb',
        write=lambda _: None,
        tell=lambda: 0,
    )
    output = omf.serializers.array_serializer(arr, open_file)
    assert output['start'] == 0
    assert output['dtype'] == dtype
    assert output['length'] == 0

def test_array_serializer_none():
    """Test array serializer when value is None"""
    open_file = mock.MagicMock(
        mode='wb',
        write=lambda _: None,
        tell=lambda: 0,
    )
    output = omf.serializers.array_serializer(None, open_file)
    assert output is None


def test_bad_shape():
    """Test bad shape for array deserializer"""
    with pytest.raises(TypeError):
        omf.serializers.array_deserializer(['*', '*'])


@mock.patch('omf.serializers.np.frombuffer')
@mock.patch('omf.serializers.zlib.decompress')
@pytest.mark.parametrize(('dtype', 'shape'), [
    ('int', [3, 3]),
    ('<i8', [5]),
    ('<i8', [10]),
    ('<i8', [4, '*']),
])
def test_bad_deserialize(mock_decompress, mock_frombuffer, dtype, shape):
    """Test expected errors during deserialization"""
    mock_decompress.return_value = None
    mock_frombuffer.return_value = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    deserializer = omf.serializers.array_deserializer(shape)
    index = {
        'dtype': dtype,
        'start': 0,
        'length': 0,
    }
    with pytest.raises(AssertionError):
        deserializer(index, io.BytesIO())


@pytest.mark.parametrize('arr', [
    np.random.randint(100, size=[4, 5, 6]),
    np.random.randint(100, size=[3]),
    np.random.rand(4, 5, 6),
    np.array([0.5, 1., np.nan, np.nan, 2.5]),
])
def test_array_serialize_round_trip(arr):
    """Test array serialization/deserialization correctly round-trips"""
    file = io.BytesIO()
    mock_file = mock.MagicMock()
    mock_file.seek = file.seek
    mock_file.read = file.read
    mock_file.tell = file.tell
    mock_file.write = file.write
    mock_file.mode = 'wb'
    for i in range(len(arr.shape) + 1):
        input_shape = list(arr.shape)
        if i > 0:
            input_shape[i-1] = '*'
        index = omf.serializers.array_serializer(arr, mock_file)
        deserializer = omf.serializers.array_deserializer(input_shape)
        output = deserializer(index, mock_file)
        assert output.shape == arr.shape
        assert output.dtype == arr.dtype
        assert np.allclose(output, arr, equal_nan=True)


def test_bad_png_serializer():
    """Test invalid PNG serializer values"""
    with pytest.raises(ValueError):
        omf.serializers.png_serializer(io.StringIO(), io.BytesIO())


def test_png_serializer():
    """Test PNG image serializes correctly"""
    output = omf.serializers.png_serializer(io.BytesIO(), io.BytesIO())
    assert output['start'] == 0
    assert output['dtype'] == 'image/png'
    assert output['length'] == 8
    assert omf.serializers.png_serializer(None, io.BytesIO()) is None


def test_bad_png_deserializer():
    """Test invalid PNG deserialier input index"""
    index = {'start': 0, 'length': 0, 'dtype': 'image'}
    with pytest.raises(AssertionError):
        omf.serializers.png_deserializer(index, io.BytesIO())


def test_png_serialize_round_trip():
    """Test PNG serialization/deserialization correctly round-trips"""
    file = io.BytesIO()
    index = omf.serializers.png_serializer(io.BytesIO(), file)
    output = omf.serializers.png_deserializer(index, file)
    assert output.tell() == 0
    file.seek(0, 2)
    assert output.tell() == 0
