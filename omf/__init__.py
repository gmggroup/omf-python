"""omf: API library for Open Mining Format file interchange format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .base import Project
from .blockmodel import RegularBlockModel, RegularSubBlockModel
from .data import (ColorArray, ColorData,
                   DateTimeArray, DateTimeColormap, DateTimeData,
                   Int2Array, Int3Array,
                   Legend, MappedData,
                   ScalarArray, ScalarColormap, ScalarData,
                   StringArray, StringData,
                   Vector2Array, Vector2Data,
                   Vector3Array, Vector3Data)
from .lineset import LineSetElement
from .pointset import PointSetElement
from .surface import SurfaceElement, SurfaceGridElement
from .texture import ImageTexture
from .volume import VolumeGridElement

from .fileio import OMFReader, OMFWriter

__version__ = '1.0.1'
__author__ = 'Global Mining Guidelines Group'
__license__ = 'MIT License'
__copyright__ = 'Copyright 2019 Global Mining Guidelines Group'
