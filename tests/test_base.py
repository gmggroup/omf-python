"""Tests for base UidModel class behaviors"""
import datetime
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
    now = datetime.datetime.utcnow()
    delta = datetime.timedelta(seconds=0.1)
    model = omf.base.UidModel()
    assert model.uid == my_id
    assert model.date_created - now < delta
    assert model.date_modified - now < delta


class MyModelWithInt(omf.base.UidModel):
    """Test class with one integer property"""
    class_type = 'my.model.with.int'
    my_int = properties.Integer('')


class MyModelWithIntAndInstance(MyModelWithInt):
    """Test class with an integer property and an instance property"""
    class_type = 'my.model.with.int.and.instance'
    my_model = properties.Instance('', omf.base.UidModel)


def test_modify():
    """Test that date_modified updates correctly on set"""
    model = MyModelWithInt()
    date_created = model.date_created
    date_modified = model.date_modified
    model.my_int = 0
    assert model.date_created == date_created
    assert model.date_modified > date_modified


def test_validate_updates():
    """Test that date_modified updates correctly on validate"""
    model = MyModelWithIntAndInstance(my_int=1)
    date_created = model.date_created
    date_modified = model.date_modified
    model.my_model = MyModelWithInt()
    model.my_model.my_int = 2
    model.validate()
    assert model.date_created == date_created
    assert model.date_modified > date_modified
    assert model.date_modified == model.my_model.date_modified


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
            'date_created': properties.DateTime.to_json(model.date_created),
            'date_modified': properties.DateTime.to_json(model.date_modified),
        }
        if isinstance(model, MyModelWithIntAndInstance):
            expected_dict.update({
                'my_int': model.my_int,
                'my_model': str(model.my_model.uid),
            })
        elif not skip_validation:
            expected_dict.update({'my_int': model.my_int})
        if include_class:
            expected_dict.update({'_type': model.class_type})
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
    dates = [datetime.datetime(2019, 1, i) for i in range(1, 5)]
    string_dates = [properties.DateTime.to_json(d) for d in dates]
    input_dict = {
        uid_a: {
            'date_created': string_dates[0],
            'date_modified': string_dates[1],
            'my_int': 0,
            'my_model': uid_b,
            'uid': uid_a,
            '_type': 'my.model.with.int.and.instance',
        },
        uid_b: {
            'date_created': string_dates[2],
            'date_modified': string_dates[3],
            'my_int': 1,
            'uid': uid_b,
            '_type': 'my.model.with.int',
        },
        '__root__': uid_a,
    }
    model_a = omf.base.UidModel.deserialize(input_dict, trusted=True)
    assert isinstance(model_a, MyModelWithIntAndInstance)
    #pylint: disable=no-member
    assert str(model_a.uid) == uid_a
    assert model_a.date_created == dates[0]
    assert model_a.date_modified == dates[1]
    assert model_a.my_int == 0
    assert isinstance(model_a.my_model, MyModelWithInt)
    assert model_a.my_model.date_created == dates[2]
    assert model_a.my_model.date_modified == dates[3]
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
