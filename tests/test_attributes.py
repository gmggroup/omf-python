"""Tests for attribute object validation"""
import datetime

import numpy as np
import properties
import pytest

import omf


# pylint: disable=W0143
def test_scalar_array():
    """Test array init and access works correctly"""
    arr = omf.attribute.Array(np.array([1, 2, 3], dtype="uint8"))
    assert arr.array.dtype.kind == "u"
    assert np.array_equal(arr.array, [1, 2, 3])
    assert arr.data_type == "Uint8Array"
    assert arr.shape == [3]
    assert arr.size == 3
    binary_dict = {}
    output = arr.serialize(include_class=False, binary_dict=binary_dict)
    assert len(binary_dict) == 1
    assert output == {
        "schema": "org.omf.v2.array.numeric",
        "data_type": "Uint8Array",
        "shape": [3],
        "size": 3,
        "array": list(binary_dict.keys())[0],
    }
    new_arr = omf.attribute.Array.deserialize(output, binary_dict=binary_dict)
    assert properties.equal(arr, new_arr)


def test_invalid_array():
    """Test Array class without valid array"""
    arr = omf.attribute.Array()
    assert arr.data_type is None
    assert arr.shape is None
    assert arr.size is None
    assert isinstance(omf.attribute.Array.deserialize(""), omf.attribute.Array)
    assert isinstance(omf.attribute.Array.deserialize({}), omf.attribute.Array)


def test_invalid_string_list():
    """Test StringList class without valid array"""
    arr = omf.attribute.StringList()
    assert arr.data_type is None
    assert arr.shape is None
    assert arr.size is None
    assert isinstance(omf.attribute.StringList.deserialize(""), omf.attribute.StringList)
    assert isinstance(omf.attribute.StringList.deserialize({}), omf.attribute.StringList)


def test_boolean_array():
    """Test boolean array bits"""
    arr = omf.attribute.Array(np.array([[1, 1], [0, 0]], dtype="bool"))
    assert arr.array.dtype.kind == "b"
    assert arr.data_type == "BooleanArray"
    assert arr.shape == [2, 2]
    assert arr.size == 1
    binary_dict = {}
    output = arr.serialize(include_class=False, binary_dict=binary_dict)
    assert len(binary_dict) == 1
    assert output == {
        "schema": "org.omf.v2.array.numeric",
        "data_type": "BooleanArray",
        "shape": [2, 2],
        "size": 1,
        "array": list(binary_dict.keys())[0],
    }
    new_arr = omf.attribute.Array.deserialize(output, binary_dict=binary_dict)
    assert properties.equal(arr, new_arr)


def test_datetime_list():
    """Test string list gives datetime data_type"""
    arr = omf.attribute.StringList(["1995-08-12T18:00:00Z", "1995-08-13T18:00:00Z"])
    assert arr.data_type == "DateTimeArray"
    assert arr.shape == [2]
    binary_dict = {}
    output = arr.serialize(include_class=False, binary_dict=binary_dict)
    assert len(binary_dict) == 1
    assert output == {
        "schema": "org.omf.v2.array.string",
        "data_type": "DateTimeArray",
        "shape": [2],
        "size": 48,
        "array": list(binary_dict.keys())[0],
    }


def test_string_list():
    """Test string list gives string data_type"""
    arr = omf.attribute.StringList.deserialize(
        {
            "shape": "",
            "data_type": "",
            "size": "",
            "array": "a",
        },
        binary_dict={"a": b'["a", "b", "c"]'},
    )
    assert arr.data_type == "StringArray"
    assert arr.shape == [3]
    assert arr.size == 15
    assert len(arr) == 3
    assert arr[0] == "a"
    assert arr[1] == "b"
    assert arr[2] == "c"
    output = arr.serialize(include_class=False)
    assert output == {
        "schema": "org.omf.v2.array.string",
        "data_type": "StringArray",
        "shape": [3],
        "size": 15,
    }


# pylint: enable=W0143


def test_array_instance_prop():
    """Test ArrayInstanceProperty validates correctly"""

    class HasArray(properties.HasProperties):
        """Test class for ArrayInstanceProperty"""

        arr = omf.attribute.ArrayInstanceProperty(
            "Array instance",
            shape=("*", 3),
            dtype=float,
        )

    harr = HasArray()
    harr.arr = np.array([[1.0, 2, 3], [4, 5, 6]])
    assert harr.validate()
    assert np.array_equal(harr.arr.array, [[1.0, 2, 3], [4, 5, 6]])
    assert harr.arr.data_type == "Float64Array"  # pylint: disable=E1101
    assert harr.arr.shape == [2, 3]
    assert harr.arr.size == 8 * 6

    with pytest.raises(properties.ValidationError):
        harr.arr = np.array([1.0, 2, 3])
    with pytest.raises(properties.ValidationError):
        harr.arr = np.array([[1, 2, 3], [4, 5, 6]])


def test_vector_data_dimensionality():
    """Test only 2D and 3D arrays are valid for vector data"""
    vattr = omf.attribute.VectorAttribute(array=[[1, 1], [2, 2], [3, 3]])
    assert vattr.array.shape == [3, 2]
    vattr = omf.attribute.VectorAttribute(array=[[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert vattr.array.shape == [3, 3]
    with pytest.raises(properties.ValidationError):
        omf.attribute.VectorAttribute(array=[1, 2, 3])
    with pytest.raises(properties.ValidationError):
        omf.attribute.VectorAttribute(array=[[1, 2, 3, 4]])


def test_contiuous_colormap():
    """Test continuous colormap validation"""
    cmap = omf.attribute.ContinuousColormap()
    with pytest.raises(properties.ValidationError):
        cmap.limits = [1.0, 0.0]
    cmap.gradient = [[0, 0, 0]] * 100
    cmap.limits = [0.0, 1.0]
    cmap.limits[0] = 2.0
    with pytest.raises(properties.ValidationError):
        cmap.validate()
    cmap.limits[0] = 0.0
    cmap.validate()
    with pytest.raises(properties.ValidationError):
        cmap.gradient = np.array([[0, 0, -1]])
    with pytest.raises(properties.ValidationError):
        cmap.gradient = np.array([[0, 0, 256]])


def test_discrete_colormap():
    """Test discrete colormap validation"""
    cmap = omf.attribute.DiscreteColormap()
    cmap.end_points = [0.5]
    cmap.end_inclusive = [True]
    cmap.colors = [[0, 100, 0], [100, 0, 0]]
    assert cmap.validate()
    cmap.end_points = [-0.5, 0.5]
    with pytest.raises(properties.ValidationError):
        cmap.validate()
    cmap.end_points = [0.5]
    cmap.end_inclusive = [True, False]
    with pytest.raises(properties.ValidationError):
        cmap.validate()
    cmap.end_inclusive = [True]
    cmap.colors = [[0, 100, 0]]
    with pytest.raises(properties.ValidationError):
        cmap.validate()
    cmap.colors = [[0, 100, 0], [100, 0, 0], [0, 0, 100]]
    with pytest.raises(properties.ValidationError):
        cmap.validate()
    cmap.end_points = [0.5, 1, 1]
    cmap.end_inclusive = [True, False, True]
    cmap.colors = [[0, 100, 0], [100, 0, 0], [0, 0, 100], [100, 100, 100]]
    assert cmap.validate()
    with pytest.raises(properties.ValidationError):
        cmap.end_points = [0.5, 1, 0.5]
    with pytest.raises(properties.ValidationError):
        cmap.end_points = [1.5, 1, 0.5]


def test_category_colormap():
    """Test legend validation"""
    legend = omf.attribute.CategoryColormap(
        name="test",
        indices=[0, 1, 2],
        values=["x", "y", "z"],
    )
    assert legend.validate()
    legend.colors = [[0, 0, 0], [0, 0, 255], [255, 0, 0]]
    assert legend.validate()
    legend = omf.attribute.CategoryColormap(
        name="test",
        indices=[0, 1, 2],
        values=["x", "y"],
    )
    with pytest.raises(properties.ValidationError):
        legend.validate()
    legend = omf.attribute.CategoryColormap(
        name="test",
        indices=[0, 1, 2],
        values=["x", "y", "z"],
        colors=[[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
    )
    with pytest.raises(properties.ValidationError):
        legend.validate()


def test_category_data():
    """Test mapped data validation"""
    mattr = omf.attribute.CategoryAttribute()
    mattr.array = [0, 2, 1, -1]
    mattr.categories = omf.attribute.CategoryColormap(
        name="letter",
        indices=[0, 1, 2],
        values=["x", "y", "z"],
    )
    mattr.location = "vertices"
    assert mattr.validate()
    with pytest.raises(properties.ValidationError):
        mattr.array = [0.5, 1.5, 2.5]
    mattr.array = [-10, 0, 1]
    assert mattr.validate()
    mattr.array.array[0] = 0
    mattr.categories.colors = ["red", "blue", "green"]
    mattr.metadata = {
        "units": "m",
        "date_created": datetime.datetime.utcnow(),
        "version": "v1.3",
    }
    assert mattr.validate()
    mattr.categories.colors = ["red", "blue"]
    with pytest.raises(properties.ValidationError):
        mattr.validate()
    with pytest.raises(properties.ValidationError):
        mattr.categories = omf.attribute.CategoryColormap(
            name="numeric",
            indices=[0, 1, 2],
            values=[0.5, 0.6, 0.7],
        )


def test_basemodel_schema():
    """Checks for a unique schema name"""
    classes_to_check = {omf.base.BaseModel}
    classes_with_subclasses = {}
    while classes_to_check:
        klass = classes_to_check.pop()
        subclasses = klass.__subclasses__()
        is_leaf = len(subclasses) == 0
        classes_with_subclasses[klass] = is_leaf
        classes_to_check |= set(subclasses)

    schemas_seen = set()
    for klass, is_leaf in classes_with_subclasses.items():
        if is_leaf:
            assert klass.schema != ""
            assert klass.schema not in schemas_seen
            schemas_seen.add(klass.schema)
        else:
            assert klass.schema == ""
