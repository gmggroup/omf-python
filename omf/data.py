"""data.py: different ProjectElementData classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

import numpy as np
import properties

from .base import UidModel, ContentModel, ProjectElementData
from .serializers import array_serializer, array_deserializer


DATA_TYPE_LOOKUP_TO_NUMPY = {
    'Int8Array': np.dtype('int8'),
    'Uint8Array': np.dtype('uint8'),
    'Int16Array': np.dtype('int16'),
    'Uint16Array': np.dtype('uint16'),
    'Int32Array': np.dtype('int32'),
    'Uint32Array': np.dtype('uint32'),
    'Int64Array': np.dtype('int64'),
    'Uint64Array': np.dtype('uint64'),
    'Float32Array': np.dtype('float32'),
    'Float64Array': np.dtype('float64'),
    'BooleanArray': np.dtype('bool'),
}
DATA_TYPE_LOOKUP_TO_STRING = {
    value: key for key, value in DATA_TYPE_LOOKUP_TO_NUMPY.items()
}


class Array(UidModel):
    """Class with unique ID and data array"""
    array = properties.Array(
        'Shared Scalar Array',
        shape={('*',), ('*', '*')},
        dtype=(int, float, bool),
        serializer=array_serializer,
        deserializer=array_deserializer,
    )

    def __init__(self, array=None, **kwargs):
        super(Array, self).__init__(**kwargs)
        if array is not None:
            self.array = array

    def __len__(self):
        return self.array.__len__()

    def __getitem__(self, i):
        return self.array.__getitem__(i)

    @properties.validator
    def _validate_datatype(self):
        if self.array.dtype not in DATA_TYPE_LOOKUP_TO_STRING:
            raise properties.ValidationError(
                'bad dtype: {} - Array must have dtype in {}'.format(
                    self.array.dtype, ', '.join(
                        [dtype.name for dtype in DATA_TYPE_LOOKUP_TO_STRING]
                    )
                )
            )
        return True

    @properties.StringChoice(
        'Array data type string', choices=list(DATA_TYPE_LOOKUP_TO_NUMPY)
    )
    def datatype(self):
        """Array type descriptor, determined directly from the array"""
        if self.array is None:
            return None
        return DATA_TYPE_LOOKUP_TO_STRING.get(self.array.dtype, None)

    @properties.List(
        'Shape of the array', properties.Integer(''),
    )
    def shape(self):
        """Array shape, determined directly from the array"""
        if self.array is None:
            return None
        return list(self.array.shape)

    @properties.Integer('Size of array in bits')
    def size(self):
        """Total size of the array in bits"""
        if self.array is None:
            return None
        bit_multiplier = 1 if self.datatype == 'BooleanArray' else 8           #pylint: disable=comparison-with-callable
        return self.array.size * self.array.itemsize * bit_multiplier


class ArrayInstanceProperty(properties.Instance):
    """Instance property for OMF Array objects

    This adds additional shape and dtype validation.
    """

    def __init__(self, doc, **kwargs):
        if 'instance_class' in kwargs:
            raise AttributeError(
                'ArrayInstanceProperty does not allow custom instance_class'
            )
        self.validator_prop = properties.Array(
            '',
            shape={('*',), ('*', '*')},
            dtype=(int, float, bool),
        )
        super(ArrayInstanceProperty, self).__init__(
            doc, instance_class=Array, **kwargs
        )

    @property
    def shape(self):
        """Required shape of the Array instance's array property"""
        return self.validator_prop.shape

    @shape.setter
    def shape(self, value):
        self.validator_prop.shape = value

    @property
    def dtype(self):
        """Required dtype of the Array instance's array property"""
        return self.validator_prop.dtype

    @dtype.setter
    def dtype(self, value):
        self.validator_prop.dtype = value


    def validate(self, instance, value):
        self.validator_prop.name = self.name
        value = super(ArrayInstanceProperty, self).validate(instance, value)
        if value.array is not None:
            value.array = self.validator_prop.validate(instance, value.array)
        return value


class StringList(UidModel):

    array = properties.Union('List of datetimes or strings',
        props=(
            properties.List('', properties.DateTime('')),
            properties.List('', properties.String('')),
        )
    )

    @properties.StringChoice(
        'List data type string', choices=['DateTimeArray', 'StringArray']
    )
    def datatype(self):
        """Array type descriptor, determined directly from the array"""
        if self.array is None:
            return None
        try:
            properties.List('', properties.DateTime('')).validate(self, self.array)
        except properties.ValidationError:
            return 'StringArray'
        return 'DateTimeArray'

    @properties.List(
        'Shape of the string list', properties.Integer(''),
    )
    def shape(self):
        """Array shape, determined directly from the array"""
        if self.array is None:
            return None
        return [len(self.array)]

    @properties.Integer('Size of string list dumped to JSON in bits')
    def size(self):
        """Total size of the string list in bits"""
        if self.array is None:
            return None
        return len(json.dumps(self.array))*8


class Colormap(ContentModel):
    """Length-128 color gradient with min/max values, used with ScalarData"""
    gradient = ArrayInstanceProperty(
        'N x 3 Array of RGB values between 0 and 255 which defines '
        'the color gradient',
        shape=('*', 3),
        dtype=int,
    )
    limits = properties.List(
        'Data range associated with the gradient',
        prop=properties.Float(''),
        min_length=2,
        max_length=2,
        default=properties.undefined,
    )

    @properties.validator('limits')
    def _check_limits_on_change(self, change):                                 #pylint: disable=no-self-use
        """Ensure limits are valid"""
        if change['value'][0] > change['value'][1]:
            raise ValueError('Colormap limits[0] must be <= limits[1]')

    @properties.validator
    def _check_limits_on_validate(self):
        """Ensure limits are valid"""
        self._check_limits_on_change({'value': self.limits})


class VectorData(ProjectElementData):
    """Data array with vector values

    This data type cannot have a colormap, since you cannot map colormaps
    to vectors.
    """
    array = ArrayInstanceProperty(
        'Numeric vectors at locations on a mesh (see location parameter); '
        'these vectors may be 2D or 3D',
        shape={('*', 2), ('*', 3)},
    )


class NumericData(ProjectElementData):
    """Data array with scalar values"""
    array = ArrayInstanceProperty(
        'Numeric values at locations on a mesh (see location parameter); '
        'these values must be scalars',
        shape=('*',),
    )
    colormap = properties.Instance(
        'colormap associated with the data',
        Colormap,
        required=False,
    )

class StringData(ProjectElementData):
    """Data consisting of a list of strings or datetimes"""
    array = properties.Instance(
        'String values at locations on a mesh (see '
        'location parameter); these values may be DateTimes or '
        'arbitrary strings',
        StringList,
    )


class Legend(ContentModel):
    """Legends to be used with MappedData indices"""
    values = properties.Union(
        'values for mapping indexed data',
        props=(
            properties.List('', properties.Color('')),
            properties.List('', properties.String('')),
            properties.List('', properties.Integer('')),
            properties.List('', properties.Float('')),
        )
    )


class MappedData(ProjectElementData):
    """Data array of indices linked to legend values or -1 for no data"""
    array = ArrayInstanceProperty(
        'indices into 1 or more legends for locations on a mesh',
        shape=('*',),
        dtype=int,

    )
    legends = properties.List(
        'legends into which the indices map',
        Legend,
        default=list,
    )

    @property
    def indices(self):
        """Allows getting/setting array with more intuitive term indices"""
        return self.array

    @indices.setter
    def indices(self, value):
        self.array = value

    def value_dict(self, i):
        """Return a dictionary of legend entries based on index"""
        if self.indices[i] == -1:
            return None
        entry = {legend.name: legend.values[self.indices[i]]                    #pylint: disable=unsubscriptable-object
                 for legend in self.legends}                                    #pylint: disable=not-an-iterable
        return entry

    @properties.validator('array')
    def _validate_min_ind(self, change):                                       #pylint: disable=no-self-use
        """This validation call fires immediately when indices is set"""
        if change['value'].array.dtype.kind != 'i':
            raise ValueError('DataMap indices must be integers')
        if np.min(change['value'].array) < -1:
            raise ValueError('DataMap indices must be >= -1')

    @properties.validator
    def _validate_indices(self):
        """This validation call fires on validate() after everything is set"""
        if np.min(self.indices.array) < -1:                                    #pylint: disable=no-member
            raise ValueError(
                'Indices of DataMap {} must be >= -1'.format(self.name)
            )
        for legend in self.legends:                                            #pylint: disable=not-an-iterable
            if np.max(self.indices.array) >= len(legend.values):               #pylint: disable=no-member
                raise ValueError(
                    'Indices of DataMap {dm} exceed number of available '
                    'entries in Legend {leg}'.format(
                        dm=self.name,
                        leg=legend.name
                    )
                )
        return True
