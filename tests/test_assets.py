"""Tests that the files in the assets folder can be read"""
import os
import pytest

import omf


class TestAssets:
    search_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))

    @classmethod
    def pytest_generate_tests(cls, metafunc):
        metafunc.parametrize('path', cls.iter_assets(), ids=cls.idfn)

    @classmethod
    def iter_assets(cls):
        for dir, _, files in os.walk(cls.search_dir):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext.lower() == '.omf':
                    _, name = os.path.split(dir)
                    yield os.path.join(dir, f)

    @classmethod
    def idfn(cls, path):
        path, name = os.path.split(path)
        _, path = os.path.split(path)
        return f'{path}.{name}'

    def test_assets(self, path):
        omf.base.BaseModel._INSTANCES = {}  # pylint: disable=W0212
        omf.load(path, include_binary=False)

        omf.base.BaseModel._INSTANCES = {}  # pylint: disable=W0212
        new_proj = omf.load(path)
        assert new_proj is not None
        assert new_proj.validate()
