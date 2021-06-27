"""base.py: OMF Project and base classes for its components"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import json
import uuid

import properties
import properties.extras
import six


class UIDMetaclass(properties.base.PropertyMetaclass):
    """Metaclass to access value from an instance registry by uid"""

    def __call__(cls, *args, **kwargs):
        """Look up an instance by uid in the registry, or make a new one"""
        if (
                len(args) == 1 and not kwargs and
                isinstance(args[0], six.string_types) and
                args[0] in cls._INSTANCES
        ):
            return cls._INSTANCES[args[0]]
        return super(UIDMetaclass, cls).__call__(*args, **kwargs)


class UidModel(six.with_metaclass(UIDMetaclass, properties.extras.HasUID)):
    """UidModel is a HasProperties object with uid"""
    _REGISTRY = OrderedDict()

    @properties.validator('uid')
    def _ensure_uuid(self, change):
        """Validate that new uids are UUID"""
        self.validate_uid(change['value'])
        return True


    @classmethod
    def validate_uid(cls, uid):
        """Validate that uid is a UUID"""
        uuid.UUID(uid)
        return True

    @classmethod
    def deserialize(cls, value, trusted=False, strict=False,
                    assert_valid=False, **kwargs):
        cls._INSTANCES = {}
        instance = super(UidModel, cls).deserialize(
            value, trusted, strict, assert_valid, **kwargs
        )
        return instance


    def serialize(self, include_class=True, save_dynamic=False, **kwargs):
        output = super(UidModel, self).serialize(
            include_class,
            save_dynamic,
            **kwargs
        )
        dict_to_mutate = None
        if isinstance(output, dict):
            dict_to_mutate = output
        elif kwargs.get('registry', None):
            dict_to_mutate = kwargs.get('registry')
        if dict_to_mutate:
            for entry in dict_to_mutate.values():
                if not isinstance(entry, dict) or '__class__' not in entry:
                    continue
                entry.update(
                    {'schema_type': self._REGISTRY[entry.pop('__class__')].schema_type}
                )
        return output


    @classmethod
    def deserialize(cls, value, trusted=False, strict=False,
                    assert_valid=False, **kwargs):
        if kwargs.get('registry', None) is None:
            if not isinstance(value, dict):
                raise ValueError('UidModel must deserialize from dictionary')
            value = value.copy()
            for entry in value.values():
                if not isinstance(entry, dict) or 'schema_type' not in entry:
                    continue
                schema_type = entry.pop('schema_type')
                for class_name, class_value in cls._REGISTRY.items():
                    if getattr(class_value, 'schema_type', '') == schema_type:
                        entry['__class__'] = class_name
                        break
                else:
                    raise ValueError(
                        'Unrecognized class type: {}'.format(schema_type)
                    )
        return super(UidModel, cls).deserialize(
            value, trusted, strict, assert_valid, **kwargs
        )


class StringDateTime(properties.DateTime):
    """DateTime property validated to be a string"""

    def validate(self, instance, value):
        value = super(StringDateTime, self).validate(instance, value)
        return self.to_json(value)


class BaseMetadata(properties.HasProperties):
    """Validated metadata properties for all objects"""
    date_created = StringDateTime(
        'Date object was created',
        required=False,
    )
    date_modified = StringDateTime(
        'Date object was modified',
        required=False,
    )


class ProjectMetadata(BaseMetadata):
    """Validated metadata properties for Projects"""
    coordinate_reference_system = properties.String(
        'EPSG or Proj4 plus optional local transformation string',
        required=False,
    )
    author = properties.String(
        'Author of the project',
        required=False,
    )
    revision = properties.String(
        'Revision',
        required=False,
    )
    date = StringDateTime(
        'Date associated with the project data',
        required=False,
    )


class ElementMetadata(BaseMetadata):
    """Validated metadata properties for Elements"""
    coordinate_reference_system = properties.String(
        'EPSG or Proj4 plus optional local transformation string',
        required=False,
    )
    color = properties.Color(
        'Solid element color',
        required=False,
    )
    opacity = properties.Float(
        'Element opacity',
        min=0,
        max=1,
        required=False,
    )


class AttributeMetadata(BaseMetadata):
    """Validated metadata properties for Attributes"""
    units = properties.String(
        'Units of attribute values',
        required=False,
    )


class ArbitraryMetadataDict(properties.Dictionary):
    """Custom property class for metadata dictionaries

    This property accepts JSON-compatible dictionary with any arbitrary
    fields. However, an additional :code:`metadata_class` is specified
    to validate specific fields.
    """

    @property
    def metadata_class(self):
        """HasProperties class to validate metadata fields against"""
        return self._metadata_class

    @metadata_class.setter
    def metadata_class(self, value):
        if not issubclass(value, properties.HasProperties):
            raise AttributeError(
                'metadata_class must be HasProperites subclass'
            )
        self._metadata_class = value                                           #pylint: disable=attribute-defined-outside-init

    def __init__(self, doc, metadata_class, **kwargs):
        self.metadata_class = metadata_class
        kwargs.update({'key_prop': properties.String('')})
        super(ArbitraryMetadataDict, self).__init__(doc, **kwargs)

    def validate(self, instance, value):
        """Validate the dictionary and any property defined in metadata_class

        This also reassigns the dictionary after validation, so any
        coerced values persist.
        """
        new_value = super(ArbitraryMetadataDict, self).validate(
            instance, value
        )
        filtered_value = properties.utils.filter_props(
            self.metadata_class,
            new_value,
        )[0]
        try:
            for key, val in filtered_value.items():
                new_value[key] = self.metadata_class._props[key].validate(
                    instance, val
                )
        except properties.ValidationError as err:
            raise properties.ValidationError(
                'Invalid metadata: {}'.format(err),
                reason='invalid',
                prop=self.name,
                instance=instance,
            )
        try:
            json.dumps(new_value)
        except TypeError:
            raise properties.ValidationError(                                  #pylint: disable=raise-missing-from
                'Metadata is not JSON compatible',
                reason='invalid',
                prop=self.name,
                instance=instance,
            )
        if not self.equal(value, new_value):
            setattr(instance, self.name, new_value)
        return value

    @property
    def info(self):
        """Description of the property, supplemental to the basic doc"""
        info = (
            'an arbitrary JSON-serializable dictionary, with certain keys '
            'validated against :class:`{cls} <{pref}.{cls}>`'.format(
                cls=self.metadata_class.__name__,
                pref=self.metadata_class.__module__,
            )
        )
        return info


class ContentModel(UidModel):
    """ContentModel is a UidModel with name, description, and metadata"""
    name = properties.String(
        'Title of the object',
        default='',
    )
    description = properties.String(
        'Description of the object',
        default='',
    )
    metadata = ArbitraryMetadataDict(
        'Basic object metadata',
        metadata_class=BaseMetadata,
        default=dict,
    )


class ProjectElementData(ContentModel):
    """Data array with values at specific locations on the mesh"""

    location = properties.StringChoice(
        'Location of the data on mesh',
        choices=('vertices', 'segments', 'faces', 'cells', 'elements'),
    )
    metadata = ArbitraryMetadataDict(
        'Attribute metadata',
        metadata_class=AttributeMetadata,
        default=dict,
    )

    @property
    def array(self):
        """Data subclasses should override array with their data array"""
        raise ValueError('Cannot access array of base ProjectElementData')


class ProjectElement(ContentModel):
    """Base ProjectElement class for OMF file

    ProjectElement subclasses must define their geometric definition.
    ProjectElements include PointSet, LineSet, Surface, and Volume
    """

    data = properties.List(
        'Data defined on the element',
        prop=ProjectElementData,
        required=False,
        default=list,
    )
    metadata = ArbitraryMetadataDict(
        'Element metadata',
        metadata_class=ElementMetadata,
        default=dict,
    )

    _valid_locations = None

    def location_length(self, location):
        """Return correct data length based on location"""
        raise NotImplementedError()

    @properties.validator
    def _validate_data(self):
        """Check if element is built correctly"""
        assert self._valid_locations, 'ProjectElement needs _valid_locations'
        for i, dat in enumerate(self.data):
            if dat.location not in self._valid_locations:                      #pylint: disable=protected-access
                raise properties.ValidationError(
                    'Invalid location {loc} - valid values: {locs}'.format(
                        loc=dat.location,
                        locs=', '.join(self._valid_locations)                  #pylint: disable=protected-access
                    )
                )
            valid_length = self.location_length(dat.location)
            if len(dat.array) != valid_length:
                raise properties.ValidationError(
                    'data[{index}] length {datalen} does not match '
                    '{loc} length {meshlen}'.format(
                        index=i,
                        datalen=len(dat.array),
                        loc=dat.location,
                        meshlen=valid_length
                    )
                )
        return True


class Project(ContentModel):
    """OMF Project for serializing to .omf file"""
    schema_type = 'org.omf.v2.project'

    elements = properties.List(
        'Project Elements',
        prop=ProjectElement,
        default=list,
    )
    metadata = ArbitraryMetadataDict(
        'Project metadata',
        metadata_class=ProjectMetadata,
        default=dict,
    )
