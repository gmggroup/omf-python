from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import subprocess
import unittest


class TestDoc(unittest.TestCase):
    @property
    def docs_dir(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        return os.path.sep.join(dirname.split(os.path.sep)[:-1] + ["docs"])

    def setUp(self):
        self.build_dir = os.path.join(self.docs_dir, "_build")
        os.makedirs(self.build_dir, exist_ok=True)

        self.doctrees_dir = os.path.join(self.build_dir, "doctrees")
        os.makedirs(self.doctrees_dir, exist_ok=True)

        self.html_dir = os.path.join(self.build_dir, "html")
        os.makedirs(self.html_dir, exist_ok=True)

    def test_html(self):
        check = subprocess.call(
            [
                "sphinx-build",
                "-nW",
                "-b",
                "html",
                "-d",
                "{}".format(self.doctrees_dir),
                "{}".format(self.docs_dir),
                "{}".format(self.html_dir),
            ]
        )
        assert check == 0

    def test_linkcheck(self):
        check = subprocess.call(
            [
                "sphinx-build",
                "-nW",
                "-b",
                "linkcheck",
                "-d",
                "{}".format(self.doctrees_dir),
                "{}".format(self.docs_dir),
                "{}".format(self.build_dir),
            ]
        )
        assert check == 0


if __name__ == "__main__":
    unittest.main()
