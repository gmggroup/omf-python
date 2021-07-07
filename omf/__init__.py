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
    Array, CategoryAttribute, CategoryColormap, ContinuousColormap,
    DiscreteColormap, NumericAttribute, StringAttribute, VectorAttribute
)
from .lineset import LineSetElement
from .pointset import PointSetElement
from .surface import SurfaceElement, SurfaceGridElement
from .texture import ProjectedTexture, UVMappedTexture

from .fileio import load, save, __version__

__author__ = 'Global Mining Guidelines Group'
__license__ = 'MIT License'
__copyright__ = 'Copyright 2019 Global Mining Guidelines Group'
