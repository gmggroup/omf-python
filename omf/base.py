"""base.py: OMF Project and base classes for its components"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import datetime
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

    date_created = properties.DateTime(
        'Date project was created',
        default=datetime.datetime.utcnow,
    )
    date_modified = properties.DateTime(
        'Date project was modified',
        default=datetime.datetime.utcnow,
    )

    @properties.observer(properties.everything)
    def _modify(self, _):
        """Update date_modified whenever anything changes"""
        self._backend['date_modified'] = datetime.datetime.utcnow()

    @properties.validator
    def _update_date_modified(self):
        """Update date_modified if any contained UidModel has been modified"""
        for val in self._backend.values():
            if (
                    isinstance(val, UidModel) and
                    val.date_modified > self.date_modified
            ):
                self._backend['date_modified'] = val.date_modified

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


class ContentModel(UidModel):
    """ContentModel is a UidModel with title and description"""
    name = properties.String(
        'Title',
        default=''
    )
    description = properties.String(
        'Description',
        default=''
    )


class ProjectElementData(ContentModel):
    """Data array with values at specific locations on the mesh"""

    location = properties.StringChoice(
        'Location of the data on mesh',
        choices=('vertices', 'segments', 'faces', 'cells')
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
    color = properties.Color(
        'Solid color',
        default='random',
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
                raise ValueError(
                    'Invalid location {loc} - valid values: {locs}'.format(
                        loc=dat.location,
                        locs=', '.join(self._valid_locations)                  #pylint: disable=protected-access
                    )
                )
            valid_length = self.location_length(dat.location)
            if len(dat.array) != valid_length:
                raise ValueError(
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
    author = properties.String(
        'Author',
        default=''
    )
    revision = properties.String(
        'Revision',
        default=''
    )
    date = properties.DateTime(
        'Date associated with the project data',
        required=False
    )
    units = properties.String(
        'Spatial units of project',
        default=''
    )
    elements = properties.List(
        'Project Elements',
        prop=ProjectElement,
        default=list,
    )
    origin = properties.Vector3(
        'Origin point for all elements in the project',
        default=[0., 0., 0.]
    )
