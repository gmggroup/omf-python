"""Tests that the files in the assets folder can be read"""
import os
import pytest

import omf


class TestAssets:

    @staticmethod
    def iter_assets():
        asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
        for dir, _, files in os.walk(asset_dir):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext.lower() == '.omf':
                    _, name = os.path.split(dir)
                    yield os.path.join(dir, f)

    @staticmethod
    def idfn(path):
        path, _ = os.path.split(path)
        _, path = os.path.split(path)
        return path

    @pytest.mark.parametrize('path', iter_assets(), ids=idfn)
    def test_assets(self, path):
        omf.base.BaseModel._INSTANCES = {}  # pylint: disable=W0212
        omf.load(path, include_binary=False)

        omf.base.BaseModel._INSTANCES = {}  # pylint: disable=W0212
        new_proj = omf.load(path)
        assert new_proj is not None
        assert new_proj.validate()
