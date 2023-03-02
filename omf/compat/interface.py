"""interface.py: Interface that all OMF readers must adhere to."""
import abc

from ..base import Project


class WrongVersionError(ValueError):
    """Raised if the initial version check failed"""


class InvalidOMFFile(ValueError):
    """Raised if loading the file failed"""


# pylint: disable=too-few-public-methods
class IOMFReader(abc.ABC):
    """Interface for readers of older OMF file versions."""

    @abc.abstractmethod
    def __init__(self, filename: str):
        pass

    @abc.abstractmethod
    def load(self, include_binary: bool = True, project_json: str = None) -> Project:
        """Attempt to load the specified file.
        See :func:`~omf.load` for parameters.
        :raises:
            WrongVersionError:
        """
