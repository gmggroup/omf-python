import abc

from ..base import Project


class WrongVersionError(ValueError):
    """Raised if the initial version check failed"""


class InvalidOMFFile(ValueError):
    """Raised if loading the file failed"""


class IOMFReader(abc.ABC):
    @abc.abstractmethod
    def __init__(self, filename: str):
        pass

    @abc.abstractmethod
    def load(self, include_binary: bool = True, project_json: str = None) -> Project:
        pass
