import os
import subprocess
import unittest


class TestDoc(unittest.TestCase):
    @property
    def docs_dir(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        return os.path.sep.join(dirname.split(os.path.sep)[:-1] + ["docs"])

    def setUp(self):
        self.build_dir = os.path.sep.join(self.docs_dir.split(os.path.sep) + ["_build"])
        if not os.path.isdir(self.build_dir):
            os.makedirs(f"{self.build_dir}")

        self.doctrees_dir = os.path.sep.join(
            self.build_dir.split(os.path.sep) + ["doctrees"]
        )
        if not os.path.isdir(self.doctrees_dir):
            os.makedirs(f"{self.doctrees_dir}")

        self.html_dir = os.path.sep.join(self.build_dir.split(os.path.sep) + ["html"])
        if not os.path.isdir(self.html_dir):
            os.makedirs(f"{self.html_dir}")

    def test_html(self):
        check = subprocess.call(
            [
                "sphinx-build",
                "-nW",
                "-b",
                "html",
                "-d",
                f"{self.doctrees_dir}",
                f"{self.docs_dir}",
                f"{self.html_dir}",
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
                f"{self.doctrees_dir}",
                f"{self.docs_dir}",
                f"{self.build_dir}",
            ]
        )
        assert check == 0


if __name__ == "__main__":
    unittest.main()
