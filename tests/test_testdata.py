"""Tests that the files in the testdata folder can be read. This folder is excluded from source control."""
import os
from . import test_assets


class TestTestdata(test_assets.TestAssets):
    """This test looks for a 'testdata' folder in the root of this repository.
    All .omf files in that folder will be loaded and validated."""

    search_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
