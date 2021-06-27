"""serializers.py: array and image serializers/deserializers for OMF file IO"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from io import BytesIO
import uuid

import numpy as np
import properties
from six import PY2

if PY2:
    memoryview = buffer                                                        #pylint: disable=redefined-builtin, invalid-name, undefined-variable

def array_serializer(arr, binary_dict=None, **kwargs):                                #pylint: disable=unused-argument
    """Convert array data to a serialized binary format"""
    if arr is None:
        return None
    if not isinstance(arr, np.ndarray):
        raise ValueError('Array must by numpy ndarray')
    arr = arr.view(np.ndarray)
    if isinstance(arr.flatten()[0], np.floating):                              #pylint: disable=no-member
        dtype = '<f8'
        nan_mask = ~np.isnan(arr)
        if not np.allclose(arr.astype(dtype)[nan_mask], arr[nan_mask]):
            raise ValueError('Converting array to float64 changed the values '
                             'of the array.')
    elif isinstance(arr.flatten()[0], np.integer):
        dtype = '<i8'
        if np.any(arr.astype(dtype) != arr):
            raise ValueError('Converting array to int64 changed the values '
                             'of the array.')
    else:
        raise ValueError('Array be float or int type: {}'.format(arr.dtype))
    uid = str(uuid.uuid4())
    index = dict()
    index['start'] = 0
    index['dtype'] = dtype
    index['array'] = uid
    index['length'] = arr.astype(dtype).nbytes
    if binary_dict is not None:
        binary_dict[uid] = arr.astype(dtype).tobytes()
    return index

def array_deserializer(index, open_file, **kwargs):                            #pylint: disable=unused-argument
    """Convert binary to numpy array based on input shape"""

    def __init__(self, shape):
        if sum([dim == '*' for dim in shape]) > 1:
            raise TypeError('array_deserializer shape may only have one '
                            'unknown dimension')
        self.shape = shape

    def __call__(self, index, binary_dict=None, **kwargs):                            #pylint: disable=unused-argument
        assert index['dtype'] in ('<i8', '<f8'), 'invalid dtype'
        if binary_dict is None or index['array'] not in binary_dict:
            return properties.undefined
        arr_buffer = binary_dict[index['array']]
        arr = np.frombuffer(arr_buffer, index['dtype'])
        unknown_dim = len(arr)
        for dim in self.shape:
            if dim == '*':
                continue
            unknown_dim /= dim
        if '*' in self.shape:
            assert abs(unknown_dim - int(unknown_dim)) < 1e-9, 'bad shape'
            shape = tuple(
                (int(unknown_dim) if dim == '*' else dim for dim in self.shape)
            )
        else:
            assert abs(unknown_dim - 1) < 1e-9, 'bad shape'
            shape = self.shape
        arr = arr.reshape(shape)
        return arr

def png_serializer(img, binary_dict=None, **kwargs):                                  #pylint: disable=unused-argument
    """Serialize PNG in bytes to file"""
    if img is None:
        return None
    if not isinstance(img, BytesIO):
        raise ValueError('Image must be BytesIO')
    uid = str(uuid.uuid4())
    index = dict()
    index['image'] = uid
    index['start'] = 0
    index['dtype'] = 'image/png'
    if binary_dict is not None:
        binary_dict[uid] = img.read()
    index['length'] = img.tell()
    return index

def png_deserializer(index, binary_dict=None, **kwargs):                              #pylint: disable=unused-argument
    """Read PNG from file as bytes"""
    assert index['dtype'] == 'image/png', 'invalid dtype'
    if binary_dict is None or index['image'] not in binary_dict:
        return properties.undefined
    img = BytesIO()
    img.write(binary_dict[index['image']])
    img.seek(0, 0)
    return img
