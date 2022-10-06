"""Tests for PointSet validation"""
import datetime
import numpy as np
import pytest
import omf


def test_pointset(tmp_path):
    """Test pointset geometry validation"""
    pts = omf.PointSetElement(
        name='Random Points',
        description='Just random points',
        geometry=omf.PointSetGeometry(
            vertices=np.random.rand(100, 3)
        ),
        data=[
            omf.ScalarData(
                name='rand data',
                array=np.random.rand(100),
                location='vertices'
            ),
        ],
        color='green'
    )

    file = str(tmp_path / "pointset.omf")
    omf.OMFWriter(pts, file)
    omf.OMFReader(file).get_project()
    pass


def test_pointset_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    pts = omf.PointSetElement(
        name='Random Points',
        description='Just random points',
        geometry=omf.PointSetGeometry(
            vertices=np.random.rand(100, 3)
        ),
        data=[
            omf.ScalarData(
                name='rand data',
                array=np.random.rand(100),
                location='vertices'
            ),
        ],
        color='green'
    )

    file = str(tmp_path / "pointset.geoh5")
    omf.OMFWriter(pts, file)

    # file = r"C:\Users\dominiquef\Documents\GIT\mira\omf\assets\LacBloom_surface.omf" # str(tmp_path / "pointset.omf")
    # new_pts: omf.PointSetElement = omf.OMFReader(file).get_project()
    #
    # assert new_pts.geometry == pts.geometry

