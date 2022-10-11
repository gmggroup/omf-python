from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

from geoh5py.data import Data
from geoh5py.objects import Curve, Points
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
from numpy import c_, ndarray, r_

from omf.base import UidModel
from omf.data import ScalarArray, ScalarData
from omf.lineset import LineSetElement, LineSetGeometry
from omf.pointset import PointSetElement, PointSetGeometry


class GeoH5Writer:
    """
    OMF to geoh5 class converter
    """

    def __init__(self, element, file_name: str | Path):
        self._file = None
        self.file = file_name
        self.entity = element

    @property
    def file(self):
        """Target file on disk."""
        return self._file

    @file.setter
    def file(self, file_name: str | Path):
        if not isinstance(file_name, (str, Path)):
            raise ValueError("Input 'file' must be of str or Path.")

        self._file = file_name

    @property
    def entity(self):
        return self._entity

    @entity.setter
    def entity(self, element):
        if type(element) not in _CLASS_MAP:
            raise ValueError(
                "Element of type {type(element)} currently not implemented."
            )

        converter = _CLASS_MAP[type(element)](element, self.file)
        self._entity = converter.from_omf()


class BaseConversion(ABC):
    """
    Base conversion between OMF and geoh5 format.
    """

    arguments: dict = {}
    geoh5: str | Path | Workspace = None
    geoh5_type = Entity
    omf_type: type[UidModel] = UidModel
    _attribute_map: dict = {
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
                f"Input object should be an instance of {self.omf_type} or {self.geoh5_type}"
            )

        self.geoh5 = geoh5

    def collect_attributes(self, **kwargs):
        with fetch_h5_handle(self.geoh5) as workspace:
            for key, alias in self._attribute_map.items():
                prop = getattr(self.element, key, None)

                if isinstance(prop, UidModel):
                    converter = _CLASS_MAP[type(prop)](prop, workspace)
                    kwargs = converter.from_omf(**kwargs)
                else:
                    kwargs[alias] = prop

        return kwargs

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
            kwargs = self.collect_attributes(**kwargs)
            self._entity = workspace.create_entity(
                self.geoh5_type, **{"entity": kwargs}
            )
            self.process_dependents(workspace)

        return self._entity

    def process_dependents(self, _: Workspace):
        """Convert children of element or entity."""

    @abstractmethod
    def from_geoh5(self) -> UidModel:
        """Convert geoh5 entity to omg element."""


class PointsConversion(BaseConversion):
    """
    Conversion between :obj:`omf.pointset.PointSetElement` and
    :obj:`geoh5py.objects.Points`
    """

    omf_type = PointSetElement
    geoh5_type = Points
    _attribute_map = {
        "description": "description",
        "name": "name",
        "uid": "uid",
        "geometry": None,
    }

    def from_geoh5(self) -> PointSetElement:
        """TODO Convert geoh5 entity to omg element."""

    def process_dependents(self, workspace):
        if getattr(self.element, "data", None):
            for child in self.element.data:
                converter = _CLASS_MAP[type(child)](child, workspace)
                converter.from_omf(parent=self.entity)


class CurveConversion(BaseConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve`
    """

    omf_type = LineSetElement
    geoh5_type = Curve
    _attribute_map = {
        "description": "description",
        "name": "name",
        "uid": "uid",
        "geometry": None,
    }

    def from_geoh5(self) -> PointSetElement:
        """TODO Convert geoh5 entity to omg element."""

    def process_dependents(self, workspace):
        if getattr(self.element, "data", None):
            for child in self.element.data:
                converter = _CLASS_MAP[type(child)](child, workspace)
                converter.from_omf(parent=self.entity)


class VerticesConversion(BaseConversion):
    """
    Conversion between :obj:`omf.pointset.PointSetGeometry` and
    :obj:`geoh5py.objects.Points.vertices`
    """

    omf_type = PointSetGeometry
    geoh5_type = ndarray
    _attribute_map: dict = {}

    def from_omf(self, **kwargs) -> dict:
        kwargs.update({"vertices": c_[self.element.vertices]})
        return kwargs

    def from_geoh5(self) -> UidModel:
        """TODO Convert geoh5 entity to omg element."""


class SegmentConversion(BaseConversion):
    """
    Conversion between :obj:`omf.lineset.LineSetElement` and
    :obj:`geoh5py.objects.Curve` `vertices` and `cells`
    """

    omf_type = LineSetGeometry
    geoh5_type = ndarray
    _attribute_map: dict = {}

    def from_omf(self, **kwargs) -> dict:
        kwargs.update(
            {"vertices": c_[self.element.vertices], "cells": c_[self.element.segments]}
        )

        return kwargs

    def from_geoh5(self) -> UidModel:
        """TODO Convert geoh5 entity to omg element."""


class DataConversion(BaseConversion):
    """
    Conversion between :obj:`omf.data.ScalarData` and
    :obj:`geoh5py.data.Data`
    """

    omf_type = ScalarData
    geoh5_type = Data
    _attribute_map = {
        "uid": "uid",
        "name": "name",
        "array": "values",
        "colormap": "color_map",
    }

    def from_omf(self, parent=None, **kwargs) -> Data:
        with fetch_h5_handle(self.geoh5):
            kwargs = self.collect_attributes(**kwargs)
            self._entity = parent.add_data({self.element.name: kwargs})

        return self._entity

    def from_geoh5(self) -> UidModel:
        """TODO Convert geoh5 entity to omg element."""


class ValuesConversion(BaseConversion):
    """
    Conversion between :obj:`omf.data.ScalarArray` and
    :obj:`geoh5py.data.Data.values`
    """

    omf_type = ScalarArray
    geoh5_type = ndarray
    _attribute_map: dict = {}

    def from_omf(self, **kwargs) -> dict:
        kwargs.update({"values": r_[self.element.array]})
        return kwargs

    def from_geoh5(self) -> UidModel:
        """TODO Convert geoh5 entity to omg element."""


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


_CLASS_MAP = {
    PointSetElement: PointsConversion,
    PointSetGeometry: VerticesConversion,
    LineSetElement: CurveConversion,
    LineSetGeometry: SegmentConversion,
    ScalarData: DataConversion,
    ScalarArray: ValuesConversion,
}
