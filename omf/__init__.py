"""omf: API library for Open Mining Format file interchange format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .base import Project
from .blockmodel import (
    ArbitrarySubBlockModel,
    OctreeSubBlockModel,
    RegularBlockModel,
    RegularSubBlockModel,
    TensorBlockModel,
)
from .composite import CompositeElement
from .data import (
    Array, CategoryData, CategoryColormap, ContinuousColormap,
    DiscreteColormap, NumericData, StringData, VectorData
)
from .lineset import LineSetElement
from .pointset import PointSetElement
from .surface import SurfaceElement, SurfaceGridElement
from .texture import ProjectedTexture, UVMappedTexture

from .fileio import OMFReader, OMFWriter

__version__ = '1.0.1'
__author__ = 'Global Mining Guidelines Group'
__license__ = 'MIT License'
__copyright__ = 'Copyright 2019 Global Mining Guidelines Group'
