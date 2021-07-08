"""attribute.py: different ProjectElementAttribute classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import uuid

import numpy as np
import properties

from .base import BaseModel, ContentModel, ProjectElementAttribute


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


class Array(BaseModel):
    """Class with unique ID and data array"""
    schema = 'org.omf.v2.array.numeric'

    array = properties.Array(
        'Shared Scalar Array',
        shape={('*',), ('*', '*')},
        dtype=(int, float, bool),
        serializer=lambda *args, **kwargs: None,
        deserializer=lambda *args, **kwargs: None,
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
    def _validate_data_type(self):
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
    def data_type(self):
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

    @properties.Integer('Size of array in bytes')
    def size(self):
        """Total size of the array in bytes"""
        if self.array is None:
            return None
        if self.data_type == 'BooleanArray':                                    #pylint: disable=comparison-with-callable
            return int(np.ceil(self.array.size / 8))
        return self.array.size * self.array.itemsize

    def serialize(self, include_class=True, save_dynamic=False, **kwargs):
        output = super(Array, self).serialize(
            include_class=include_class, save_dynamic=True, **kwargs
        )
        binary_dict = kwargs.get('binary_dict', None)
        if binary_dict is not None:
            array_uid = str(uuid.uuid4())
            if self.data_type == 'BooleanArray':                                #pylint: disable=comparison-with-callable
                array_binary = np.packbits(self.array, axis=None).tobytes()
            else:
                array_binary = self.array.tobytes()
            binary_dict.update({array_uid: array_binary})
            output.update({'array': array_uid})
        return output

    @classmethod
    def deserialize(cls, value, trusted=False, strict=False,
                    assert_valid=False, **kwargs):
        binary_dict = kwargs.get('binary_dict', {})
        if not isinstance(value, dict):
            pass
        elif any(key not in value for key in ['shape', 'data_type', 'array']):
            pass
        elif value['array'] in binary_dict:
            array_binary = binary_dict[value['array']]
            array_dtype = DATA_TYPE_LOOKUP_TO_NUMPY[value['data_type']]
            if value['data_type'] == 'BooleanArray':
                int_arr = np.frombuffer(array_binary, dtype='uint8')
                bit_arr = np.unpackbits(int_arr)[:np.product(value['shape'])]
                arr = bit_arr.astype(array_dtype)
            else:
                arr = np.frombuffer(array_binary, dtype=array_dtype)
            arr = arr.reshape(value['shape'])
            return cls(arr)
        return cls()

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


class StringList(BaseModel):
    """Array-like class with unique ID and string-list array"""
    schema = 'org.omf.v2.array.string'

    array = properties.List('List of datetimes or strings',
        properties.String(''),
        serializer=lambda *args, **kwargs: None,
        deserializer=lambda *args, **kwargs: None,
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
    def data_type(self):
        """Array type descriptor, determined directly from the array"""
        if self.array is None:
            return None
        try:
            properties.List('', properties.DateTime('')).validate(
                self, self.array
            )
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

    @properties.Integer('Size of string list dumped to JSON in bytes')
    def size(self):
        """Total size of the string list in bytes"""
        if self.array is None:
            return None
        return len(json.dumps(self.array))

    def serialize(self, include_class=True, save_dynamic=False, **kwargs):
        output = super(StringList, self).serialize(
            include_class=include_class, save_dynamic=True, **kwargs
        )
        binary_dict = kwargs.get('binary_dict', None)
        if binary_dict is not None:
            array_uid = str(uuid.uuid4())
            binary_dict.update(
                {array_uid: bytes(json.dumps(self.array), 'utf8')}
            )
            output.update({'array': array_uid})
        return output

    @classmethod
    def deserialize(cls, value, trusted=False, strict=False,
                    assert_valid=False, **kwargs):
        binary_dict = kwargs.get('binary_dict', {})
        if not isinstance(value, dict):
            pass
        elif any(key not in value for key in ['shape', 'data_type', 'array']):
            pass
        elif value['array'] in binary_dict:
            arr = json.loads(binary_dict[value['array']].decode('utf8'))
            return cls(arr)
        return cls()


class ContinuousColormap(ContentModel):
    """Color gradient with min/max values, used with NumericAttribute"""
    schema = 'org.omf.v2.colormap.scalar'

    gradient = ArrayInstanceProperty(
        'N x 3 Array of RGB values between 0 and 255 which defines '
        'the color gradient',
        shape=('*', 3),
        dtype=int,
    )
    limits = properties.List(
        'Attribute range associated with the gradient',
        prop=properties.Float(''),
        min_length=2,
        max_length=2,
        default=properties.undefined,
    )

    @properties.validator('gradient')
    def _check_gradient_values(self, change):                                  #pylint: disable=no-self-use
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

class DiscreteColormap(ContentModel):
    """Colormap for grouping discrete intervals of NumericAttribute"""

    schema = 'org.omf.v2.colormap.discrete'

    end_points = properties.List(
        'Attribute values associated with edge of color intervals',
        prop=properties.Float(''),
        default=properties.undefined,
    )
    end_inclusive = properties.List(
        'True if corresponding end_point is included in lower interval; '
        'False if end_point is in upper interval',
        prop=properties.Boolean(''),
        default=properties.undefined,
    )
    colors = properties.List(
        'Colors for each interval',
        prop=properties.Color(''),
        min_length=1,
        default=properties.undefined,
    )

    @properties.validator
    def _validate_lengths(self):
        if len(self.end_points) != len(self.end_inclusive):
            pass
        elif len(self.colors) == len(self.end_points) + 1:
            return True
        raise properties.ValidationError(
            'Discrete colormap colors length must be one greater than '
            'end_points and end_inclusive values'
        )

    @properties.validator('end_points')
    def _validate_end_points_monotonic(self, change):                          #pylint: disable=no-self-use
        for i in range(len(change['value']) - 1):
            diff = change['value'][i+1] - change['value'][i]
            if diff < 0:
                raise properties.ValidationError(
                    'end_points must be monotonically increasing'
                )



class NumericAttribute(ProjectElementAttribute):
    """Attribute array with scalar values"""
    schema = 'org.omf.v2.attribute.numeric'

    array = ArrayInstanceProperty(
        'Numeric values at locations on a mesh (see location parameter); '
        'these values must be scalars',
        shape=('*',),
    )
    colormap = properties.Union(
        'colormap associated with the attribute',
        [ContinuousColormap, DiscreteColormap],
        required=False,
    )


class VectorAttribute(ProjectElementAttribute):
    """Attribute array with vector values

    This Attribute type cannot have a colormap, since you cannot map colormaps
    to vectors.
    """
    schema = 'org.omf.v2.attribute.vector'

    array = ArrayInstanceProperty(
        'Numeric vectors at locations on a mesh (see location parameter); '
        'these vectors may be 2D or 3D',
        shape={('*', 2), ('*', 3)},
    )

class StringAttribute(ProjectElementAttribute):
    """Attribute consisting of a list of strings or datetimes"""
    schema = 'org.omf.v2.attribute.string'

    array = properties.Instance(
        'String values at locations on a mesh (see '
        'location parameter); these values may be DateTimes or '
        'arbitrary strings',
        StringList,
    )


class CategoryColormap(ContentModel):
    """Legends to be used with CategoryAttribute indices"""
    schema = 'org.omf.v2.colormap.category'

    indices = properties.List(
        'indices corresponding to CateogryAttribute array values',
        properties.Integer(''),
    )
    values = properties.List(
        'values for mapping indexed attribute',
        properties.String(''),
    )
    colors = properties.List(
        'colors corresponding to values',
        properties.Color(''),
        required=False,
    )

    @properties.validator
    def _validate_lengths(self):
        if len(self.indices) != len(self.values):
            pass
        elif self.colors is None or len(self.colors) == len(self.values):
            return True
        raise properties.ValidationError(
            'Legend colors and values must be the same length'
        )


class CategoryAttribute(ProjectElementAttribute):
    """Attribute array of indices linked to category values

    For no attribute, indices should correspond to a value outside the
    range of the categories.
    """
    schema = 'org.omf.v2.attribute.category'

    array = ArrayInstanceProperty(
        'indices into the category values for locations on a mesh',
        shape=('*',),
        dtype=int,

    )
    categories = properties.Instance(
        'categories into which the indices map',
        CategoryColormap,
    )
