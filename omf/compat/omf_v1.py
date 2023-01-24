import json
import struct
import uuid

from .interface import IOMFReader, InvalidOMFFile


COMPATIBILITY_VERSION = b'OMF-v0.9.0'


class Reader(IOMFReader):
    def __init__(self, filename: str):
        self._filename = filename

    def load(self, include_binary: bool = True, project_json: bool = None):
        with open(self._filename, 'rb') as f:
            project_uuid, json_start = self._read_header(f)
            project_json = self._read_json(f, json_start)

        raise InvalidOMFFile


    def _read_header(self, f):
        """Checks magic number and version; gets project uid and json start"""
        f.seek(0)
        if f.read(4) != b'\x84\x83\x82\x81':
            raise InvalidOMFFile(f'Unsupported format: {self._filename}')
        file_version = struct.unpack('<32s', f.read(32))[0]
        file_version = file_version[0:len(COMPATIBILITY_VERSION)]
        if file_version != COMPATIBILITY_VERSION:
            raise InvalidOMFFile("Unsupported file version: {}".format(file_version))
        project_uuid = uuid.UUID(bytes=struct.unpack('<16s', f.read(16))[0])
        json_start = struct.unpack('<Q', f.read(8))[0]
        return str(project_uuid), json_start

    def _read_json(self, f, json_start):
        f.seek(json_start, 0)
        return json.loads(f.read().decode('utf-8'))