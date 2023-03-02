"""omf: API library for Open Mining Format file interchange format"""
from .attribute import (
    Array,
    CategoryAttribute,
    CategoryColormap,
    ContinuousColormap,
    DiscreteColormap,
    NumericAttribute,
    StringAttribute,
    VectorAttribute,
)
from .base import Project
from .blockmodel import (
    FreeformSubblockedModel,
    OctreeSubblockDefinition,
    RegularBlockModel,
    RegularBlockModelDefinition,
    RegularSubblockDefinition,
    SubblockedModel,
    TensorBlockModelDefinition,
    TensorGridBlockModel,
)
from .composite import Composite
from .fileio import __version__, load, save
from .lineset import LineSet
from .pointset import PointSet
from .surface import Surface, TensorGridSurface
from .texture import ProjectedTexture, UVMappedTexture

__author__ = "Global Mining Guidelines Group"
__license__ = "MIT License"
__copyright__ = "Copyright 2021 Global Mining Guidelines Group"
