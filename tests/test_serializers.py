"""Tests for file serializers"""
import io
try:
    from unittest import mock
except ImportError:
    import mock

import numpy as np
import pytest

import omf


@pytest.mark.parametrize('arr', [
    [1, 2, 3],
    np.array(['a', 'b', 'c']),
])
def test_bad_array_serializer(arr):
    """Test array serializer failures"""
    with pytest.raises(ValueError):
        omf.serializers.array_serializer(arr, {})


@pytest.mark.parametrize(('arr', 'dtype'), [
    (np.array([1., 2., 3.]), '<f8'),
    (np.array([np.nan, 2., 3.]), '<f8'),
    (np.array([1, 2, 3]), '<i8'),
])
def test_array_serializer(arr, dtype):
    """Test array serializer creates correct output index"""
    binary_dict = {}
    output = omf.serializers.array_serializer(arr, binary_dict)
    assert output['start'] == 0
    assert output['dtype'] == dtype
    assert output['length'] == 24
    assert output['array'] in binary_dict

def test_array_serializer_none():
    """Test array serializer when value is None"""
    output = omf.serializers.array_serializer(None, {})
    assert output is None


def test_bad_shape():
    """Test bad shape for array deserializer"""
    with pytest.raises(TypeError):
        omf.serializers.array_deserializer(['*', '*'])


@mock.patch('omf.serializers.np.frombuffer')
@pytest.mark.parametrize(('dtype', 'shape'), [
    ('int', [3, 3]),
    ('<i8', [5]),
    ('<i8', [10]),
    ('<i8', [4, '*']),
])
def test_bad_deserialize(mock_frombuffer, dtype, shape):
    """Test expected errors during deserialization"""
    mock_frombuffer.return_value = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    deserializer = omf.serializers.array_deserializer(shape)
    index = {
        'dtype': dtype,
        'start': 0,
        'length': 0,
        'array': 'a',
    }
    with pytest.raises(AssertionError):
        deserializer(index, {'a': None})


@pytest.mark.parametrize('arr', [
    np.random.randint(100, size=[4, 5, 6]),
    np.random.randint(100, size=[3]),
    np.random.rand(4, 5, 6),
    np.array([0.5, 1., np.nan, np.nan, 2.5]),
])
def test_array_serialize_round_trip(arr):
    """Test array serialization/deserialization correctly round-trips"""
    for i in range(len(arr.shape) + 1):
        input_shape = list(arr.shape)
        if i > 0:
            input_shape[i-1] = '*'
        binary_dict = {}
        index = omf.serializers.array_serializer(arr, binary_dict)
        deserializer = omf.serializers.array_deserializer(input_shape)
        output = deserializer(index, binary_dict)
        assert output.shape == arr.shape
        assert output.dtype == arr.dtype
        assert np.allclose(output, arr, equal_nan=True)


def test_bad_png_serializer():
    """Test invalid PNG serializer values"""
    with pytest.raises(ValueError):
        omf.serializers.png_serializer(io.StringIO(), {})


def test_png_serializer():
    """Test PNG image serializes correctly"""
    binary_dict = {}
    output = omf.serializers.png_serializer(io.BytesIO(), binary_dict)
    assert output['start'] == 0
    assert output['dtype'] == 'image/png'
    assert output['length'] == 0
    assert output['image'] in binary_dict
    assert omf.serializers.png_serializer(None, {}) is None


def test_bad_png_deserializer():
    """Test invalid PNG deserialier input index"""
    index = {'start': 0, 'length': 0, 'dtype': 'image'}
    with pytest.raises(AssertionError):
        omf.serializers.png_deserializer(index, {})


def test_png_serialize_round_trip():
    """Test PNG serialization/deserialization correctly round-trips"""
    binary_dict = {}
    index = omf.serializers.png_serializer(io.BytesIO(), binary_dict)
    output = omf.serializers.png_deserializer(index, binary_dict)
    assert output.tell() == 0
