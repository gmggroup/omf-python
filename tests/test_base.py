"""Tests for BaseModel class behaviors"""
import datetime
import json

import numpy as np
import properties
import properties.extras
import pytest

import omf


class Metadata(properties.HasProperties):
    """Metadata class for testing"""

    meta_int = properties.Integer("", required=False)
    meta_string = properties.String("", required=False)
    meta_color = properties.Color("", required=False)
    meta_anything = properties.Property("", required=False)
    meta_date = omf.base.StringDateTime("", required=False)


def test_metadata_property():
    """Test metadata validates predefined keys but allows any keys"""

    class WithMetadata(properties.HasProperties):
        """Test class with metadata"""

        metadata = omf.base.ArbitraryMetadataDict(
            "Some metadata",
            Metadata,
            default=dict,
        )

    with pytest.raises(AttributeError):
        WithMetadata._props["metadata"].metadata_class = object  # pylint: disable=E1101

    has_metadata = WithMetadata()
    assert has_metadata.validate()
    has_metadata.metadata["meta_int"] = 5
    assert has_metadata.validate()
    has_metadata.metadata["meta_int"] = "not an int"
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata["meta_int"] = 5
    has_metadata.metadata["meta_string"] = "a string"
    assert has_metadata.validate()
    has_metadata.metadata["meta_color"] = "red"
    assert has_metadata.validate()
    assert has_metadata.metadata["meta_color"] == (255, 0, 0)
    has_metadata.metadata["meta_anything"] = "a string"
    assert has_metadata.validate()
    has_metadata.metadata["meta_anything"] = Metadata
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata["meta_anything"] = "a string"
    has_metadata.metadata["meta_date"] = "some date"
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata["meta_date"] = datetime.datetime(1980, 1, 1)
    assert has_metadata.validate()
    has_metadata.metadata["another"] = "a string"
    has_metadata.metadata["even another"] = "a string"
    assert has_metadata.validate()

    has_metadata.metadata["and another"] = Metadata
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata.pop("and another")
    serialized_has_meta = has_metadata.serialize(include_class=False)
    assert serialized_has_meta == {
        "metadata": {
            "meta_int": 5,
            "meta_string": "a string",
            "meta_color": (255, 0, 0),
            "meta_anything": "a string",
            "meta_date": "1980-01-01T00:00:00Z",
            "another": "a string",
            "even another": "a string",
        }
    }
    new_metadata = WithMetadata.deserialize(json.loads(json.dumps(serialized_has_meta)))
    new_metadata.validate()
    assert properties.equal(has_metadata, new_metadata)
    assert new_metadata.serialize(include_class=False) == serialized_has_meta


class MyModelWithInt(omf.base.BaseModel):
    """Test class with one integer property"""

    schema = "my.model.with.int"
    my_int = properties.Integer("")


class MyModelWithIntAndInstance(MyModelWithInt):
    """Test class with an integer property and an instance property"""

    schema = "my.model.with.int.and.instance"
    my_model = properties.Instance("", omf.base.BaseModel)


@pytest.mark.parametrize("include_class", [True, False])
def test_uid_model_serialize(include_class):
    """Test BaseModel correctly serializes to flat dictionary"""
    model = MyModelWithIntAndInstance(
        my_int=0,
        my_model=MyModelWithIntAndInstance(
            my_int=1,
            my_model=MyModelWithInt(),
        ),
    )
    expected = {
        "schema": "my.model.with.int.and.instance",
        "my_int": 0,
        "my_model": {
            "schema": "my.model.with.int.and.instance",
            "my_int": 1,
            "my_model": {
                "schema": "my.model.with.int",
            },
        },
    }
    if include_class:
        expected["__class__"] = "MyModelWithIntAndInstance"
        expected["my_model"]["__class__"] = "MyModelWithIntAndInstance"
        expected["my_model"]["my_model"]["__class__"] = "MyModelWithInt"
    assert model.serialize(include_class=include_class) == expected


def test_deserialize():
    """Test deserialize correctly builds BaseModel from registry"""
    input_dict = {
        "my_int": 0,
        "my_model": {
            "my_int": 1,
            "schema": "my.model.with.int",
        },
        "schema": "my.model.with.int.and.instance",
    }
    model_a = omf.base.BaseModel.deserialize(input_dict, trusted=True)
    assert isinstance(model_a, MyModelWithIntAndInstance)
    # pylint: disable=E1101
    assert model_a.my_int == 0
    assert isinstance(model_a.my_model, MyModelWithInt)
    assert model_a.my_model.my_int == 1
    # pylint: enable=E1101


class MockArray(omf.base.BaseModel):
    """Test array class"""

    array = np.array([1, 2, 3])


class MockAttribute(omf.base.ProjectElementAttribute):
    """Test attribute class"""

    array = MockArray()


def test_project_element():
    """Test validation of element geometry and attributes"""
    element = omf.base.ProjectElement()
    with pytest.raises(AssertionError):
        element.validate()
    element._valid_locations = ("vertices",)  # pylint: disable=W0212
    element.location_length = lambda _: 5
    element.attributes = [MockAttribute(location="faces")]
    with pytest.raises(ValueError):
        element.validate()
    element.attributes = [MockAttribute(location="vertices")]
    with pytest.raises(ValueError):
        element.validate()
    element.location_length = lambda _: 3
    assert element.validate()
