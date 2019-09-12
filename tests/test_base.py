"""Tests for base UidModel class behaviors"""
import datetime
import json
try:
    from unittest import mock
except ImportError:
    import mock
import uuid

import properties
import properties.extras
import pytest

import omf


@mock.patch('properties.extras.uid.uuid')
def test_uid_model(mock_uuid):
    """Test UidModel default behaviour"""
    my_id = str(uuid.uuid4())
    mock_uuid.UUID = uuid.UUID
    mock_uuid.uuid4 = lambda: my_id
    model = omf.base.UidModel()
    assert model.uid == my_id


class Metadata(properties.HasProperties):
    """Metadata class for testing"""
    meta_int = properties.Integer('', required=False)
    meta_string = properties.String('', required=False)
    meta_color = properties.Color('', required=False)
    meta_anything = properties.Property('', required=False)
    meta_date = omf.base.StringDateTime('', required=False)


def test_metadata_property():
    """Test metadata validates predefined keys but allows any keys"""

    class WithMetadata(properties.HasProperties):
        """Test class with metadata"""
        metadata = omf.base.ArbitraryMetadataDict(
            'Some metadata',
            Metadata,
            default=dict,
        )

    with pytest.raises(AttributeError):
        WithMetadata._props['metadata'].metadata_class = object                #pylint: disable=no-member

    has_metadata = WithMetadata()
    assert has_metadata.validate()
    has_metadata.metadata['meta_int'] = 5
    assert has_metadata.validate()
    has_metadata.metadata['meta_int'] = 'not an int'
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata['meta_int'] = 5
    has_metadata.metadata['meta_string'] = 'a string'
    assert has_metadata.validate()
    has_metadata.metadata['meta_color'] = 'red'
    assert has_metadata.validate()
    assert has_metadata.metadata['meta_color'] == (255, 0, 0)
    has_metadata.metadata['meta_anything'] = 'a string'
    assert has_metadata.validate()
    has_metadata.metadata['meta_anything'] = Metadata
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata['meta_anything'] = 'a string'
    has_metadata.metadata['meta_date'] = 'some date'
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata['meta_date'] = datetime.datetime(1980, 1, 1)
    assert has_metadata.validate()
    has_metadata.metadata['another'] = 'a string'
    has_metadata.metadata['even another'] = 'a string'
    assert has_metadata.validate()

    has_metadata.metadata['and another'] = Metadata
    with pytest.raises(properties.ValidationError):
        has_metadata.validate()
    has_metadata.metadata.pop('and another')
    serialized_has_meta = has_metadata.serialize(include_class=False)
    assert serialized_has_meta == {
        'metadata': {
            'meta_int': 5,
            'meta_string': 'a string',
            'meta_color': (255, 0, 0),
            'meta_anything': 'a string',
            'meta_date': '1980-01-01T00:00:00Z',
            'another': 'a string',
            'even another': 'a string',
        }
    }
    new_metadata = WithMetadata.deserialize(
        json.loads(json.dumps(serialized_has_meta))
    )
    new_metadata.validate()
    assert properties.equal(has_metadata, new_metadata)
    assert new_metadata.serialize(include_class=False) == serialized_has_meta


class MyModelWithInt(omf.base.UidModel):
    """Test class with one integer property"""
    my_int = properties.Integer('')


class MyModelWithIntAndInstance(MyModelWithInt):
    """Test class with an integer property and an instance property"""
    my_model = properties.Instance('', omf.base.UidModel)


@pytest.mark.parametrize('include_class', [True, False])
@pytest.mark.parametrize('skip_validation', [True, False])
@pytest.mark.parametrize('registry', [None, {'key': 'value'}])
def test_uid_model_serialize(include_class, skip_validation, registry):
    """Test UidModel correctly serializes to flat dictionary"""
    model = MyModelWithIntAndInstance(
        my_int=0,
        my_model=MyModelWithIntAndInstance(
            my_int=1,
            my_model=MyModelWithInt(),
        ),
    )
    if not skip_validation:
        model.my_model.my_model.my_int = 2
    output_registry = registry.copy() if registry else None
    output = model.serialize(
        include_class=include_class,
        skip_validation=skip_validation,
        registry=output_registry,
    )
    if registry:
        assert output == str(model.uid)
        assert 'key' in output_registry
        assert output_registry['key'] == 'value'
        output_registry.pop('key')
        output = output_registry
        assert len(output) == 3
    else:
        assert len(output) == 4
    for model in [model, model.my_model, model.my_model.my_model]:
        assert model.uid in output
        expected_dict = {
            'uid': model.uid,
        }
        if isinstance(model, MyModelWithIntAndInstance):
            expected_dict.update({
                'my_int': model.my_int,
                'my_model': str(model.my_model.uid),
            })
        elif not skip_validation:
            expected_dict.update({'my_int': model.my_int})
        if include_class:
            expected_dict.update({'__class__': model.__class__.__name__})
        assert output[str(model.uid)] == expected_dict


@pytest.mark.parametrize('registry', [
    {'__root__': 'my_int', 'my_model': 1}, None
])
def test_bad_deserialize(registry):
    """Test deserialize fails with bad registry"""
    with pytest.raises(ValueError):
        omf.base.UidModel.deserialize(registry, trusted=True)


def test_deserialize():
    """Test deserialize correctly builds UidModel from registry"""
    uid_a = str(uuid.uuid4())
    uid_b = str(uuid.uuid4())
    input_dict = {
        uid_a: {
            'my_int': 0,
            'my_model': uid_b,
            'uid': uid_a,
            '__class__': 'MyModelWithIntAndInstance',
        },
        uid_b: {
            'my_int': 1,
            'uid': uid_b,
            '__class__': 'MyModelWithInt',
        },
        '__root__': uid_a,
    }
    model_a = omf.base.UidModel.deserialize(input_dict, trusted=True)
    assert isinstance(model_a, MyModelWithIntAndInstance)
    #pylint: disable=no-member
    assert str(model_a.uid) == uid_a
    assert model_a.my_int == 0
    assert isinstance(model_a.my_model, MyModelWithInt)
    assert model_a.my_model.my_int == 1
    input_dict['__root__'] = uid_b
    properties.extras.HasUID._INSTANCES = {}                                   #pylint: disable=protected-access
    model_b = omf.base.UidModel.deserialize(input_dict, trusted=True)
    assert properties.equal(model_b, model_a.my_model)
    #pylint: enable=no-member


class MockData(omf.base.ProjectElementData):
    """Test data class"""
    array = [1, 2, 3]


def test_project_element():
    """Test validation of element geometry and data"""
    element = omf.base.ProjectElement()
    with pytest.raises(AssertionError):
        element.validate()
    element._valid_locations = ('vertices',)                                   #pylint: disable=protected-access
    element.location_length = lambda _: 5
    element.data = [MockData(location='faces')]
    with pytest.raises(ValueError):
        element.validate()
    element.data = [MockData(location='vertices')]
    with pytest.raises(ValueError):
        element.validate()
    element.location_length = lambda _: 3
    assert element.validate()
