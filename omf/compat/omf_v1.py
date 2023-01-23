from .interface import IOMFReader, InvalidOMFFile


class Reader(IOMFReader):
    def __init__(self, filename: str):
        self._filename = filename

    def load(self, include_binary: bool = True, project_json: bool = None):
        raise InvalidOMFFile
