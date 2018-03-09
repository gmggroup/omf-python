"""base.py: OMF Project and base classes for its components"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime

import properties


class UidModel(properties.HasProperties):
    """UidModel is a HasProperties object with uid"""
    uid = properties.Uuid(
        'Unique identifier',
        serializer=lambda val, **kwargs: None,
        deserializer=lambda val, **kwargs: None
    )
    date_created = properties.GettableProperty(
        'Date project was created',
        default=datetime.datetime.utcnow,
        serializer=properties.DateTime.to_json,
        deserializer=lambda val, **kwargs: None
    )
    date_modified = properties.GettableProperty(
        'Date project was modified',
        default=datetime.datetime.utcnow,
        serializer=properties.DateTime.to_json,
        deserializer=lambda val, **kwargs: None
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

    def serialize(self, include_class=True, registry=None,                     #pylint: disable=arguments-differ
                  skip_validation=False, **kwargs):
        """Serialize nested UidModels to a flat dictionary with pointers"""
        if registry is None:
            if not skip_validation:
                self.validate()
            registry = dict()
            root = True
        else:
            root = False
        if str(self.uid) not in registry:
            registry.update({
                str(self.uid): super(UidModel, self).serialize(
                    include_class, registry=registry, **kwargs
                )
            })
        if root:
            return registry
        return str(self.uid)

    @classmethod
    def deserialize(cls, uid, trusted=True, registry=None, **kwargs):          #pylint: disable=arguments-differ
        """Deserialize nested UidModels from flat pointer dictionary"""
        if registry is None:
            raise ValueError('no registry provided')
        if uid not in registry:
            raise ValueError('uid not found: {}'.format(uid))
        if not isinstance(registry[uid], UidModel):
            date_created = registry[uid]['date_created']
            date_modified = registry[uid]['date_modified']
            kwargs.update({'verbose': False})
            new_model = super(UidModel, cls).deserialize(
                value=registry[uid],
                registry=registry,
                trusted=trusted,
                **kwargs
            )
            new_model._backend.update({
                'uid': properties.Uuid.from_json(uid),
                'date_created': properties.DateTime.from_json(date_created),
                'date_modified': properties.DateTime.from_json(date_modified)
            })
            registry.update({uid: new_model})
        return registry[uid]


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


class ProjectElementGeometry(UidModel):
    """Base class for all ProjectElement meshes"""

    _valid_locations = None

    origin = properties.Vector3(
        'Origin of the Mesh relative to origin of the Project',
        default=[0., 0., 0.]
    )

    def location_length(self, location):
        """Return correct data length based on location"""
        raise NotImplementedError()

    @property
    def num_nodes(self):
        """get number of nodes"""
        raise NotImplementedError()

    @property
    def num_cells(self):
        """get number of cells"""
        raise NotImplementedError()


class ProjectElement(ContentModel):
    """Base ProjectElement class for OMF file

    ProjectElement subclasses must define their mesh.
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
        default='random'
    )
    geometry = None

    @properties.validator
    def _validate_data(self):
        """Check if element is built correctly"""
        assert self.geometry is not None, 'ProjectElement must have a mesh'
        for i, dat in enumerate(self.data):
            if dat.location not in self.geometry._valid_locations:             #pylint: disable=protected-access
                raise ValueError(
                    'Invalid location {loc} - valid values: {locs}'.format(
                        loc=dat.location,
                        locs=', '.join(self.geometry._valid_locations)         #pylint: disable=protected-access
                    )
                )
            valid_length = self.geometry.location_length(dat.location)
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
