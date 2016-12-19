"""fileio.py: OMF Writer and Reader for serializing to and from .omf files"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import struct
import uuid

from six import string_types

from .base import UidModel

__version__ = b'OMF-v0.9.0'


class OMFWriter(object):
    """OMFWriter serializes a OMF project to a file

    .. code::

        proj = omf.project()
        ...
        omf.OMFWriter(proj, 'outfile.omf')

    The output file starts with a 60 byte header:

    * 4 byte magic number: :code:`b'\\x81\\x82\\x83\\x84'`
    * 32 byte version string: :code:`'OMF-v0.9.0'` (other bytes empty)
    * 16 byte project uid (in little-endian bytes)
    * 8 byte unsigned long long (little-endian): JSON start location in file

    Following the header is a binary data blob.

    Following the binary is a UTF-8 encoded JSON dictionary containing
    all elements of the project keyed by UID string. Objects can reference
    each other by UID, and arrays and images contain pointers to their data
    in the binary blob.
    """

    def __init__(self, project, fname):
        """Project serialization is performed on OMFWriter init

        Binary data is written during project serialization
        """
        if len(fname) < 4 or fname[-4:] != '.omf':
            fname = fname + '.omf'
        self.fname = fname
        with open(fname, 'wb') as fopen:
            self.initialize_header(fopen, project.uid)
            self.project_json = project.serialize(open_file=fopen)
            self.update_header(fopen)
            fopen.write(json.dumps(self.project_json).encode('utf-8'))

    @staticmethod
    def initialize_header(fopen, uid):
        """Write magic number, version string, project uid, and zero bytes

        Total header length = 60 bytes

        4 (magic number)
        + 32 (version)
        + 16 (uid in bytes)
        + 8 (JSON start, written later)
        """
        fopen.seek(0, 0)
        fopen.write(b'\x84\x83\x82\x81')
        fopen.write(struct.pack('<32s', __version__.ljust(32, b'\x00')))
        fopen.write(struct.pack('<16s', uid.bytes))
        fopen.seek(8, 1)

    @staticmethod
    def update_header(fopen):
        """Return to header and write the correct JSON start location"""
        json_start = fopen.tell()
        fopen.seek(52, 0)
        fopen.write(struct.pack('<Q', json_start))
        fopen.seek(json_start)


class OMFReader(object):
    """OMFReader takes a filename and returns an OMF project

    .. code::

        proj = omf.OMFReader('infile.omf')

    """

    def __new__(cls, fopen):
        """Project serialization is performed on OMFWriter init"""
        if isinstance(fopen, string_types):
            fopen = open(fopen, 'rb')
            opened_on_start = True
        else:
            opened_on_start = False
        fopen.seek(0, 0)
        uid, json_start = cls.read_header(fopen)
        project_json = cls.read_json(fopen, json_start)
        project = UidModel.deserialize(
            uid=uid, registry=project_json, open_file=fopen
        )
        if opened_on_start:
            fopen.close()
        return project

    @staticmethod
    def read_header(fopen):
        """Checks magic number and version; gets project uid and json start"""
        if fopen.read(4) != b'\x84\x83\x82\x81':
            raise ValueError('Invalid OMF file')
        file_version = struct.unpack('<32s', fopen.read(32))[0]
        file_version = file_version[0:len(__version__)]
        if file_version != __version__:
            raise ValueError(
                'Version mismatch: file version {fv}, '
                'reader version {rv}'.format(
                    fv=file_version,
                    rv=__version__
                )
            )
        uid = str(uuid.UUID(bytes=struct.unpack('<16s', fopen.read(16))[0]))
        json_start = struct.unpack('<Q', fopen.read(8))[0]
        return uid, json_start

    @staticmethod
    def read_json(fopen, json_start):
        """Gets json dictionary from utf-8 encoded string"""
        fopen.seek(json_start, 0)
        return json.loads(fopen.read().decode('utf-8'))
