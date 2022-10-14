import os
import unittest

import numpy as np

import omf


class TestDocEx(unittest.TestCase):
    @staticmethod
    def make_random_project():
        dirname, _ = os.path.split(os.path.abspath(__file__))
        pngfile = os.path.sep.join(
            dirname.split(os.path.sep)[:-1] + ["docs", "images", "PointSetGeometry.png"]
        )
        proj = omf.Project(
            name="Test project", description="Just some assorted elements"
        )

        pts = omf.PointSetElement(
            name="Random Points",
            description="Just random points",
            geometry=omf.PointSetGeometry(vertices=np.random.rand(100, 3)),
            data=[
                omf.ScalarData(
                    name="rand data", array=np.random.rand(100), location="vertices"
                ),
                omf.ScalarData(
                    name="More rand data",
                    array=np.random.rand(100),
                    location="vertices",
                ),
            ],
            textures=[
                omf.ImageTexture(
                    name="test image",
                    image=pngfile,
                    origin=[0, 0, 0],
                    axis_u=[1, 0, 0],
                    axis_v=[0, 1, 0],
                ),
                omf.ImageTexture(
                    name="test image",
                    image=pngfile,
                    origin=[0, 0, 0],
                    axis_u=[1, 0, 0],
                    axis_v=[0, 0, 1],
                ),
            ],
            color="green",
        )

        lin = omf.LineSetElement(
            name="Random Line",
            geometry=omf.LineSetGeometry(
                vertices=np.random.rand(100, 3),
                segments=np.floor(np.random.rand(50, 2) * 100).astype(int),
            ),
            data=[
                omf.ScalarData(
                    name="rand vert data",
                    array=np.random.rand(100),
                    location="vertices",
                ),
                omf.ScalarData(
                    name="rand segment data",
                    array=np.random.rand(50),
                    location="segments",
                ),
            ],
            color="#0000FF",
        )

        surf = omf.SurfaceElement(
            name="trisurf",
            geometry=omf.SurfaceGeometry(
                vertices=np.random.rand(100, 3),
                triangles=np.floor(np.random.rand(50, 3) * 100).astype(int),
            ),
            data=[
                omf.ScalarData(
                    name="rand vert data",
                    array=np.random.rand(100),
                    location="vertices",
                ),
                omf.ScalarData(
                    name="rand face data", array=np.random.rand(50), location="faces"
                ),
            ],
            color=[100, 200, 200],
        )

        grid = omf.SurfaceElement(
            name="gridsurf",
            geometry=omf.SurfaceGridGeometry(
                tensor_u=np.ones(10).astype(float),
                tensor_v=np.ones(15).astype(float),
                origin=[50.0, 50.0, 50.0],
                axis_u=[1.0, 0, 0],
                axis_v=[0, 0, 1.0],
                offset_w=np.random.rand(11, 16).flatten(),
            ),
            data=[
                omf.ScalarData(
                    name="rand vert data",
                    array=np.random.rand(11, 16).flatten(),
                    location="vertices",
                ),
                omf.ScalarData(
                    name="rand face data",
                    array=np.random.rand(10, 15).flatten(order="f"),
                    location="faces",
                ),
            ],
            textures=[
                omf.ImageTexture(
                    name="test image",
                    image=pngfile,
                    origin=[2.0, 2.0, 2.0],
                    axis_u=[5.0, 0, 0],
                    axis_v=[0, 2.0, 5.0],
                )
            ],
        )

        vol = omf.VolumeElement(
            name="vol",
            geometry=omf.VolumeGridGeometry(
                tensor_u=np.ones(10).astype(float),
                tensor_v=np.ones(15).astype(float),
                tensor_w=np.ones(20).astype(float),
                origin=[10.0, 10.0, -10],
            ),
            data=[
                omf.ScalarData(
                    name="Random Data",
                    location="cells",
                    array=np.random.rand(10, 15, 20).flatten(),
                )
            ],
        )

        proj.elements = [pts, lin, surf, grid, vol]

        return proj


def test_doc_ex(tmp_path):
    proj = TestDocEx.make_random_project()

    assert proj.validate()

    serial_file = str(tmp_path / "out.omf")
    omf.OMFWriter(proj, serial_file)
    reader = omf.OMFReader(serial_file)
    new_proj = reader.get_project()

    assert new_proj.validate()
    assert str(new_proj.elements[3].textures[0].uid) == str(
        proj.elements[3].textures[0].uid
    )
    del reader
    os.remove(serial_file)
