"""Tests for texture validation"""
import os
import numpy as np
import png
import properties
import pytest

import omf


def setup_texture(func):
    """Function wrapper to create png image"""

    def new_func():
        """Create png image and pass to func"""
        dirname, _ = os.path.split(os.path.abspath(__file__))
        pngfile = os.path.sep.join([dirname, "out.png"])
        img = ["110010010011", "101011010100", "110010110101", "100010010011"]
        img = [[int(val) for val in value] for value in img]
        writer = png.Writer(len(img[0]), len(img), greyscale=True, bitdepth=16)
        with open(pngfile, "wb") as file:
            writer.write(file, img)
        try:
            func(pngfile)
        finally:
            os.remove(pngfile)

    return new_func


@setup_texture
def test_projectedtexture(pngfile):
    """Test projected texture validation"""
    tex = omf.ProjectedTexture()
    tex.image = pngfile
    assert tex.validate()


@setup_texture
def test_uvmappedtexture(pngfile):
    """Test uv mapped texture validation"""
    tex = omf.UVMappedTexture()
    tex.image = pngfile
    with pytest.raises(properties.ValidationError):
        tex.uv_coordinates = [0.0, 1.0, 0.5]
    tex.uv_coordinates = [[0.0, -0.5], [0.5, 1]]
    assert tex.validate()
    tex.uv_coordinates = [[0.0, 0.5], [0.5, np.nan]]
    assert tex.validate()

    points = omf.PointSet()
    points.vertices = [[0.0, 0, 0], [1, 1, 1], [2, 2, 2]]
    points.textures = [tex]
    with pytest.raises(properties.ValidationError):
        points.validate()
    points.vertices = [[0.0, 0, 0], [1, 1, 1]]
    assert points.validate()
