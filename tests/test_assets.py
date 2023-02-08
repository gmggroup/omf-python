"""Tests that the files in the assets folder can be read"""
import os

import omf


class TestAssets:
    """Tests that the files in the assets folder can be read"""

    search_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

    @classmethod
    def pytest_generate_tests(cls, metafunc):  # pylint: disable=missing-function-docstring
        metafunc.parametrize("path", cls.iter_assets(), ids=cls.idfn)

    @classmethod
    def iter_assets(cls):
        """Yields the full path of all omf files inside cls.search_dir"""
        for dir_, _, files in os.walk(cls.search_dir):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() == ".omf":
                    yield os.path.join(dir_, filename)

    @classmethod
    def idfn(cls, path):
        """Generates a test-name from a given filename"""
        if not isinstance(path, str):
            return "test"
        path, name = os.path.split(path)
        _, path = os.path.split(path)
        return f"{path}.{name}"

    def test_assets(self, path):
        """Tests that the file can be loaded with/without binary data"""
        omf.base.BaseModel._INSTANCES = {}  # pylint: disable=W0212
        omf.load(path, include_binary=False)

        omf.base.BaseModel._INSTANCES = {}  # pylint: disable=W0212
        new_proj = omf.load(path)
        assert new_proj is not None
        assert new_proj.validate()
