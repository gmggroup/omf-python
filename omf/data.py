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
    schema_type = 'org.omf.v2.array.numeric'

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
    """Array-like class with unique ID and string-list array"""
    schema_type = 'org.omf.v2.array.string'

    array = properties.Union('List of datetimes or strings',
        props=(
            properties.List('', properties.DateTime('')),
            properties.List('', properties.String('')),
        )
    )

    def __init__(self, array=None, **kwargs):
        super(StringList, self).__init__(**kwargs)
        if array is not None:
            self.array = array

    def __len__(self):
        return self.array.__len__()

    def __getitem__(self, i):
        return self.array.__getitem__(i)

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
    """Color gradient with min/max values, used with NumericData"""
    schema_type = 'org.omf.v2.colormap.scalar'

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

    @properties.validator('gradient')
    def _check_gradient_values(self, change):
        """Ensure gradient values are all between 0 and 255"""
        arr = change['value'].array
        if arr is None:
            return
        arr_uint8 = arr.astype('uint8')
        if not np.array_equal(arr, arr_uint8):
            raise properties.ValidationError(
                'Gradient must be an array of RGB values between 0 and 255'
            )
        change['value'].array = arr_uint8

    @properties.validator('limits')
    def _check_limits_on_change(self, change):                                 #pylint: disable=no-self-use
        """Ensure limits are valid"""
        if change['value'][0] > change['value'][1]:
            raise properties.ValidationError(
                'Colormap limits[0] must be <= limits[1]'
            )


class VectorData(ProjectElementData):
    """Data array with vector values

    This data type cannot have a colormap, since you cannot map colormaps
    to vectors.
    """
    schema_type = 'org.omf.v2.data.vector'

    array = ArrayInstanceProperty(
        'Numeric vectors at locations on a mesh (see location parameter); '
        'these vectors may be 2D or 3D',
        shape={('*', 2), ('*', 3)},
    )


class NumericData(ProjectElementData):
    """Data array with scalar values"""
    schema_type = 'org.omf.v2.data.numeric'

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
    schema_type = 'org.omf.v2.data.string'

    array = properties.Instance(
        'String values at locations on a mesh (see '
        'location parameter); these values may be DateTimes or '
        'arbitrary strings',
        StringList,
    )


class Legend(ContentModel):
    """Legends to be used with CategoryData indices"""
    schema_type = 'org.omf.v2.legend'

    values = properties.List(
        'values for mapping indexed data',
        properties.String(''),
    )
    colors = properties.List(
        'colors corresponding to values',
        properties.Color(''),
        required=False,
    )

    @properties.validator
    def _validate_lengths(self):
        if self.colors is None or len(self.colors) == len(self.values):
            return True
        raise properties.ValidationError(
            'Legend colors and values must be the same length'
        )


class CategoryData(ProjectElementData):
    """Data array of indices linked to category values

    For no data, indices should correspond to a value outside the
    range of the categories.
    """
    schema_type = 'org.omf.v2.data.category'

    array = ArrayInstanceProperty(
        'indices into the category values for locations on a mesh',
        shape=('*',),
        dtype=int,

    )
    categories = properties.Instance(
        'categories into which the indices map',
        Legend,
    )

    @property
    def indices(self):
        """Allows getting/setting array with more intuitive term indices"""
        return self.array

    @indices.setter
    def indices(self, value):
        self.array = value
