"""omf: API library for Open Mining Format file interchange format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .base import Project
from .data import (ColorArray, ColorData,
                   DateTimeArray, DateTimeColormap, DateTimeData,
                   Int2Array, Int3Array,
                   Legend, MappedData,
                   ScalarArray, ScalarColormap, ScalarData,
                   StringArray, StringData,
                   Vector2Array, Vector2Data,
                   Vector3Array, Vector3Data)
from .lineset import LineSetGeometry, LineSetElement
from .pointset import PointSetGeometry, PointSetElement
from .surface import SurfaceGeometry, SurfaceGridGeometry, SurfaceElement
from .texture import ImageTexture
from .volume import VolumeGridGeometry, VolumeElement

from .fileio import OMFReader, OMFWriter

__version__ = '1.0.0'
__author__ = 'Global Mining Guidelines Group'
__license__ = 'MIT License'
__copyright__ = 'Copyright 2019 Global Mining Guidelines Group'
