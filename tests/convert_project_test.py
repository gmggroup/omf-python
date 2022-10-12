"""Tests for PointSet validation"""

from geoh5py.workspace import Workspace

import omf

from .doc_example_test import TestDocEx


def test_project_to_geoh5(tmp_path):
    """Test pointset geometry validation"""
    proj = TestDocEx.make_random_project()

    file = str(tmp_path / "block_model.geoh5")
    omf.OMFWriter(proj, file)

    with Workspace(file) as workspace:
        assert len(workspace.objects) == len(proj.elements) - 1
