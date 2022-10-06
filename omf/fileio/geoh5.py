import warnings

from abc import ABC, abstractmethod

from numpy import ndarray, c_, r_

from omf.base import UidModel
from omf.data import ScalarData, ScalarArray
from omf.pointset import PointSetElement, PointSetGeometry
from contextlib import contextmanager
from geoh5py.workspace import Workspace
from geoh5py.data import Data
from geoh5py.objects import Points
from geoh5py.shared import Entity
from pathlib import Path


class GeoH5Writer(object):
    """
    OMF to geoh5 class converter
    """
    def __init__(self, element, file_name):
        self._file: str | None = None
        self.file: str = file_name
        self.entity = element

    @property
    def file(self):
        """Target file on disk."""
        return self._file

    @file.setter
    def file(self, file_name: str | Path):
        if not isinstance(file_name, str | Path):
            raise ValueError("Input 'file' must be of str or Path.")

        self._file = file_name

    @property
    def entity(self):
        return self._entity

    @entity.setter
    def entity(self, element):
        if type(element) not in MAP:
            raise ValueError("Element of type {type(element)} currently not implemented.")

        converter = MAP[type(element)](element, self.file)
        self._entity = converter.from_omf()


class BaseConversion(ABC):
    arguments = {}
    geoh5 = None
    geoh5_type = Entity
    omf_type = UidModel
    _attribute_map = {
        "uid": "uid",
        "description": "description",
        "name": "name",
    }
    _element = None
    _entity = None

    def __init__(self, obj: PointSetElement | Points, geoh5: str | Path | Workspace):
        if isinstance(obj, self.omf_type):
            self.element = obj
        elif isinstance(obj, self.geoh5_type):
            self.entity = obj
        else:
            raise ValueError(
                f"Input object should be an ininstance of {self.omf_type} or {self.geoh5_type}"
            )

        self.geoh5 = geoh5

    @property
    def element(self):
        if self._element is None and self._entity is not None:
            self.from_geoh5()
        return self._element

    @element.setter
    def element(self, value: UidModel):
        if not isinstance(value, self.omf_type):
            raise ValueError(f"Input 'element' must be of type {self.omf_type}")
        self._element = value

    @property
    def entity(self):
        if self._entity is None and self._element is not None:
            self.from_omf()
        return self._entity

    @entity.setter
    def entity(self, value: Entity):
        if not isinstance(value, self.geoh5_type):
            raise ValueError(f"Input 'entity' must be of type {self.omf_type}")
        self._entity = value

    def from_omf(self, **kwargs) -> Entity:
        """Convert omg element to geoh5 entity."""
        with fetch_h5_handle(self.geoh5) as workspace:

            for key, alias in self._attribute_map.items():
                prop = getattr(self.element, key, None)

                if isinstance(prop, UidModel):
                    converter = MAP[type(prop)](prop, workspace)
                    prop = converter.from_omf()

                kwargs[alias] = prop

            self._entity = workspace.create_entity(self.geoh5_type, **{"entity": kwargs})
            self.process_dependents(workspace)

    def process_dependents(self, workspace):
        ...

    @abstractmethod
    def from_geoh5(self) -> UidModel:
        """Convert geoh5 entity to omg element."""
        ...


class PointsConversion(BaseConversion):
    """"""
    omf_type = PointSetElement
    geoh5_type = Points
    _attribute_map = {
        "description": "description",
        "name": "name",
        "uid": "uid",
        "geometry": "vertices",
    }

    def from_geoh5(self) -> PointSetElement:
        """Convert geoh5 entity to omg element."""
        ...

    def process_dependents(self, workspace):
        if getattr(self.element, "data", None):
            for child in self.element.data:
                converter = MAP[type(child)](child, workspace)
                converter.from_omf(parent=self.entity)


class VerticesConversion(BaseConversion):
    """"""
    omf_type = PointSetGeometry
    geoh5_type = ndarray
    _attribute_map = {}

    def from_omf(self, **kwargs) -> ndarray:
        return c_[self.element.vertices]

    def from_geoh5(self) -> UidModel:
        ...


class DataConversion(BaseConversion):
    """"""
    omf_type = ScalarData
    geoh5_type = Data
    _attribute_map = {
        "uid": "uid",
        "name": "name",
        "array": "values",
        "colormap": "color_map"
    }

    def from_omf(self, parent=None, **kwargs):
        with fetch_h5_handle(self.geoh5) as workspace:

            for key, alias in self._attribute_map.items():
                prop = getattr(self.element, key, None)
                if isinstance(prop, UidModel):
                    converter = MAP[type(prop)](prop, workspace)
                    prop = converter.from_omf()

                kwargs[alias] = prop

            self._entity = parent.add_data({self.element.name: kwargs})

    def from_geoh5(self) -> UidModel:
        ...


class ValuesConversion(BaseConversion):
    omf_type = ScalarArray
    geoh5_type = ndarray
    _attribute_map = {}

    def from_omf(self, parent=None, **kwargs) -> ndarray:
        return r_[self.element.array]

    def from_geoh5(self) -> UidModel:
        ...


@contextmanager
def fetch_h5_handle(file: str | Workspace | Path, mode: str = "a") -> Workspace:
    """
    Open in read+ mode a geoh5 file from string.
    If receiving a file instead of a string, merely return the given file.

    :param file: Name or handle to a geoh5 file.
    :param mode: Set the h5 read/write mode

    :return h5py.File: Handle to an opened h5py file.
    """
    if isinstance(file, Workspace):
        try:
            yield file
        finally:
            pass
    else:
        if Path(file).suffix != ".geoh5":
            raise ValueError("Input h5 file must have a 'geoh5' extension.")

        h5file = Workspace(file, mode)

        try:
            yield h5file
        finally:
            h5file.close()


MAP = {
    PointSetElement: PointsConversion,
    PointSetGeometry: VerticesConversion,
    ScalarData: DataConversion,
    ScalarArray: ValuesConversion
}
