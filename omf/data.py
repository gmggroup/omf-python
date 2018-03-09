"""data.py: different ProjectElementData classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties

from .base import UidModel, ContentModel, ProjectElementData
from .serializers import array_serializer, array_deserializer


class ScalarArray(UidModel):
    """Class with unique ID and data array"""
    array = properties.Array(
        'Shared Scalar Array',
        serializer=array_serializer,
        deserializer=array_deserializer(('*',))
    )

    def __init__(self, array=None, **kwargs):
        super(ScalarArray, self).__init__(**kwargs)
        if array is not None:
            self.array = array

    def __len__(self):
        return self.array.__len__()

    def __getitem__(self, i):
        return self.array.__getitem__(i)


class Vector2Array(ScalarArray):
    """Shared array of 2D vectors"""
    array = properties.Vector2Array(
        'Shared Vector2 Array',
        serializer=array_serializer,
        deserializer=array_deserializer(('*', 2))
    )


class Vector3Array(ScalarArray):
    """Shared array of 3D vectors"""
    array = properties.Vector3Array(
        'Shared Vector3 Array',
        serializer=array_serializer,
        deserializer=array_deserializer(('*', 3))
    )


class Int2Array(ScalarArray):
    """Shared n x 2 array of integers"""
    array = properties.Array(
        'Shared n x 2 Int Array',
        dtype=int,
        shape=('*', 2),
        serializer=array_serializer,
        deserializer=array_deserializer(('*', 2))
    )


class Int3Array(ScalarArray):
    """Shared n x 3 array of integers"""
    array = properties.Array(
        'Shared n x 3 Int Array',
        dtype=int,
        shape=('*', 3),
        serializer=array_serializer,
        deserializer=array_deserializer(('*', 3))
    )


class StringArray(ScalarArray):
    """Shared array of text strings"""
    array = properties.List(
        'Shared array of text strings',
        prop=properties.String(''),
        default=list,
    )


class DateTimeArray(ScalarArray):
    """Shared array of DateTimes"""
    array = properties.List(
        'Shared array of DateTimes',
        prop=properties.DateTime(''),
        default=list,
    )


class ColorArray(ScalarArray):
    """Shared array of Colors"""
    array = properties.List(
        'Shared array of Colors',
        prop=properties.Color(''),
        default=list,
    )


class ScalarColormap(ContentModel):
    """Length-128 color gradient with min/max values, used with ScalarData"""
    gradient = properties.Instance(
        'length-128 ColorArray defining the gradient',
        ColorArray
    )
    limits = properties.List(
        'Data range associated with the gradient',
        prop=properties.Float(''),
        min_length=2,
        max_length=2,
        default=properties.undefined,
    )

    @properties.validator('gradient')
    def _check_gradient_length(self, change):                                  #pylint: disable=no-self-use
        """Ensure gradient is length-128"""
        if len(change['value']) != 128:
            raise ValueError('Colormap gradient must be length 128')

    @properties.validator('limits')
    def _check_limits_on_change(self, change):                                 #pylint: disable=no-self-use
        """Ensure limits are valid"""
        if change['value'][0] > change['value'][1]:
            raise ValueError('Colormap limits[0] must be <= limits[1]')

    @properties.validator
    def _check_limits_on_validate(self):
        """Ensure limits are valid"""
        self._check_limits_on_change({'value': self.limits})


class DateTimeColormap(ScalarColormap):
    """Length-128 color gradient with min/max values, used with DateTimeData"""
    limits = properties.List(
        'Data range associated with the gradient',
        prop=properties.DateTime(''),
        min_length=2,
        max_length=2,
        default=properties.undefined,
    )


class ScalarData(ProjectElementData):
    """Data array with scalar values"""
    array = properties.Instance(
        'scalar values at locations on a mesh (see location parameter)',
        ScalarArray
    )
    colormap = properties.Instance(
        'colormap associated with the data',
        ScalarColormap,
        required=False
    )


class Vector3Data(ProjectElementData):
    """Data array with 3D vectors"""
    array = properties.Instance(
        '3D vectors at locations on a mesh (see location parameter)',
        Vector3Array
    )


class Vector2Data(ProjectElementData):
    """Data array with 2D vectors"""
    array = properties.Instance(
        '2D vectors at locations on a mesh (see location parameter)',
        Vector2Array
    )


class ColorData(ProjectElementData):
    """Data array of RGB colors specified as three integers 0-255 or color

    If n x 3 integers is provided, these will simply be clipped to values
    between 0 and 255 inclusive; invalid colors will not error. This
    allows fast array validation rather than slow element-by-element list
    validation.

    Other color formats may be used (ie String or Hex colors). However,
    for large arrays, validation of these types will be slow.
    """
    array = properties.Union(
        'RGB color values at locations on a mesh (see location parameter)',
        props=(
            Int3Array,
            ColorArray
        )
    )

    @properties.validator('array')
    def _clip_colors(self, change):                                            #pylint: disable=no-self-use
        """This validation call fires immediately when array is set"""
        if isinstance(change['value'], Int3Array):
            change['value'].array = np.clip(change['value'].array, 0, 255)


class StringData(ProjectElementData):
    """Data array with text entries"""
    array = properties.Instance(
        'text at locations on a mesh (see location parameter)',
        StringArray
    )


class DateTimeData(ProjectElementData):
    """Data array with DateTime entries"""
    array = properties.Instance(
        'datetimes at locations on a mesh (see location parameter)',
        DateTimeArray
    )
    colormap = properties.Instance(
        'colormap associated with the data',
        DateTimeColormap,
        required=False
    )


class Legend(ContentModel):
    """Legends to be used with DataMap indices"""
    values = properties.Union(
        'values for mapping indexed data',
        props=(
            ColorArray,
            DateTimeArray,
            StringArray,
            ScalarArray
        )
    )


class MappedData(ProjectElementData):
    """Data array of indices linked to legend values or -1 for no data"""
    array = properties.Instance(
        'indices into 1 or more legends for locations on a mesh',
        ScalarArray
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
