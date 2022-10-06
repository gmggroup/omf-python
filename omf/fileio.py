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
    """OMFReader deserializes an OMF file.

    .. code::

        # Read all elements
        reader = omf.OMFReader('infile.omf')
        project = reader.get_project()

        # Read all PointSets:
        reader = omf.OMFReader('infile.omf')
        project = reader.get_project_overview()
        uids_to_import = [element.uid for element in project.elements
                          if isinstance(element, omf.PointSetElement)]
        filtered_project = reader.get_project(uids_to_import)

    """

    def __init__(self, fopen):
        if isinstance(fopen, string_types):
            fopen = open(fopen, 'rb')
        self._fopen = fopen
        fopen.seek(0, 0)
        self._uid, self._json_start = self.read_header()
        self._project_json = self.read_json()

    def __del__(self):
        self._fopen.close()

    def get_project(self, element_uids=None):
        """Fully loads project elements.
        Elements can be filtered by specifying their UUIDs.

        :param element_uids: a list of element UUIDs to load, default: all
        :return: a omf.base.Project containing the specified elements
        """
        project_json = self._project_json.copy()
        if element_uids is not None:
            project_elements = project_json[self._uid]
            # update the root element list
            filtered_elements = [uid for uid in project_elements['elements']
                                 if uid in element_uids]
            project_elements['elements'] = filtered_elements

        project = UidModel.deserialize(uid=self._uid,
                                       registry=project_json,
                                       open_file=self._fopen)
        return project

    def get_project_overview(self):
        """Loads all project elements without loading their data.

        :return: a omf.base.Project
        """
        project_elements = self._project_json[self._uid]
        element_uids = project_elements['elements']
        filtered_json = {self._uid: project_elements}
        for uid in element_uids:
            element = self._project_json[uid].copy()
            for prop in ('data', 'geometry', 'textures'):
                if prop in element:
                    del element[prop]
            filtered_json[uid] = element
        project = UidModel.deserialize(uid=self._uid,
                                       registry=filtered_json,
                                       open_file=self._fopen)
        return project

    def read_header(self):
        """Checks magic number and version; gets project uid and json start"""
        if self._fopen.read(4) != b'\x84\x83\x82\x81':
            raise ValueError('Invalid OMF file')
        file_version = struct.unpack('<32s', self._fopen.read(32))[0]
        file_version = file_version[0:len(__version__)]
        if file_version != __version__:
            raise ValueError(
                'Version mismatch: file version {fv}, '
                'reader version {rv}'.format(
                    fv=file_version,
                    rv=__version__
                )
            )
        uid = uuid.UUID(bytes=struct.unpack('<16s', self._fopen.read(16))[0])
        json_start = struct.unpack('<Q', self._fopen.read(8))[0]
        return str(uid), json_start

    def read_json(self):
        """Gets json dictionary from project file"""
        self._fopen.seek(self._json_start, 0)
        return json.loads(self._fopen.read().decode('utf-8'))
