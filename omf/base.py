"""base.py: OMF Project and base classes for its components"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

import properties
import properties.extras


class BaseModel(properties.HasProperties):
    """BaseModel is a HasProperties object with schema_type"""

    schema_type = ''

    def serialize(self, include_class=True, save_dynamic=False, **kwargs):
        output = super(BaseModel, self).serialize(
            include_class,
            save_dynamic,
            **kwargs
        )
        output.update({'schema_type': self.schema_type})
        return output


    @classmethod
    def deserialize(cls, value, trusted=False, strict=False,
                    assert_valid=False, **kwargs):
        schema_type = value.pop('schema_type', '')
        for class_name, class_value in cls._REGISTRY.items():
            if not hasattr(class_value, 'schema_type'):
                continue
            if class_value.schema_type == schema_type:
                value.update({'__class__': class_name})
                break
        return super(BaseModel, cls).deserialize(
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


class ContentModel(BaseModel):
    """ContentModel is a model with name, description, and metadata"""
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


class ProjectElementAttribute(ContentModel):
    """Attribute array with values at specific locations on the mesh"""

    location = properties.StringChoice(
        'Location of the attribute on mesh',
        choices=('vertices', 'segments', 'faces', 'cells', 'elements'),
    )
    metadata = ArbitraryMetadataDict(
        'Attribute metadata',
        metadata_class=AttributeMetadata,
        default=dict,
    )

    @property
    def array(self):
        """Attribute subclasses should override array"""
        raise ValueError('Cannot access array of base ProjectElementAttribute')


class ProjectElement(ContentModel):
    """Base ProjectElement class for OMF file

    ProjectElement subclasses must define their geometric definition.
    ProjectElements include PointSet, LineSet, Surface, and Volume
    """

    attributes = properties.List(
        'Attributes defined on the element',
        prop=ProjectElementAttribute,
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
        """Return correct attribute length based on location"""
        raise NotImplementedError()

    @properties.validator
    def _validate_attributes(self):
        """Check if element is built correctly"""
        assert self._valid_locations, 'ProjectElement needs _valid_locations'
        for i, attr in enumerate(self.attributes):
            if attr.location not in self._valid_locations:                      #pylint: disable=protected-access
                raise properties.ValidationError(
                    'Invalid location {loc} - valid values: {locs}'.format(
                        loc=attr.location,
                        locs=', '.join(self._valid_locations)                  #pylint: disable=protected-access
                    )
                )
            valid_length = self.location_length(attr.location)
            if len(attr.array.array) != valid_length:
                raise properties.ValidationError(
                    'attributes[{index}] length {attrlen} does not match '
                    '{loc} length {meshlen}'.format(
                        index=i,
                        attrlen=len(attr.array.array),
                        loc=attr.location,
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
