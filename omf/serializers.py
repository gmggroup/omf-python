"""serializers.py: array and image serializers/deserializers for OMF file IO"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from io import BytesIO
import zlib

import numpy as np
from six import PY2

if PY2:
    memoryview = buffer                                                        #pylint: disable=redefined-builtin, invalid-name, undefined-variable

def array_serializer(arr, open_file, **kwargs):                                #pylint: disable=unused-argument
    """Convert array data to a serialized binary format"""
    if arr is None:
        return None
    if open_file.mode != 'wb':
        raise ValueError('file mode must be wb')
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
    index = dict()
    index['start'] = open_file.tell()
    index['dtype'] = dtype
    arr_buffer = memoryview(arr.astype(dtype))
    open_file.write(zlib.compress(arr_buffer))
    index['length'] = open_file.tell() - index['start']
    index['shape'] = arr.shape
    return index

def array_deserializer(index, open_file, **kwargs):                            #pylint: disable=unused-argument
    """Convert binary to numpy array based on input shape"""
    assert index['dtype'] in ('<i8', '<f8'), 'invalid dtype'
    open_file.seek(index['start'], 0)
    arr_buffer = zlib.decompress(open_file.read(index['length']))
    arr = np.frombuffer(arr_buffer, index['dtype'])
    arr = arr.reshape(index['shape'])
    return arr

def png_serializer(img, open_file, **kwargs):                                  #pylint: disable=unused-argument
    """Serialize PNG in bytes to file"""
    if img is None:
        return None
    if not isinstance(img, BytesIO):
        raise ValueError('Image must be BytesIO')
    index = dict()
    index['start'] = open_file.tell()
    index['dtype'] = 'image/png'
    img.seek(0, 0)
    open_file.write(zlib.compress(img.read()))
    index['length'] = open_file.tell() - index['start']
    return index

def png_deserializer(index, open_file, **kwargs):                              #pylint: disable=unused-argument
    """Read PNG from file as bytes"""
    assert index['dtype'] == 'image/png', 'invalid dtype'
    open_file.seek(index['start'], 0)
    img = BytesIO()
    img.write(zlib.decompress(open_file.read(index['length'])))
    img.seek(0, 0)
    return img
