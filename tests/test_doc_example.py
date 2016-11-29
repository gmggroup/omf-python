from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

import numpy as np
import omf


class TestDocEx(unittest.TestCase):

    def test_doc_ex(self):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        pngfile = os.path.sep.join(dirname.split(os.path.sep)[:-1] +
                                   ['docs', 'images', 'PointSetGeometry.png'])

        proj = omf.Project(
            name='Test project',
            description='Just some assorted elements'
        )

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
                omf.ScalarData(
                    name='More rand data',
                    array=np.random.rand(100),
                    location='vertices'
                )
            ],
            textures=[
                omf.ImageTexture(
                    name='test image',
                    image=pngfile,
                    origin=[0, 0, 0],
                    axis_u=[1, 0, 0],
                    axis_v=[0, 1, 0]
                ),
                omf.ImageTexture(
                    name='test image',
                    image=pngfile,
                    origin=[0, 0, 0],
                    axis_u=[1, 0, 0],
                    axis_v=[0, 0, 1]
                )
            ],
            color='green'
        )

        lin = omf.LineSetElement(
            name='Random Line',
            geometry=omf.LineSetGeometry(
                vertices=np.random.rand(100, 3),
                segments=np.floor(np.random.rand(50, 2)*100).astype(int)
            ),
            data=[
                omf.ScalarData(
                    name='rand vert data',
                    array=np.random.rand(100),
                    location='vertices'
                ),
                omf.ScalarData(
                    name='rand segment data',
                    array=np.random.rand(50),
                    location='segments'
                )
            ],
            color='#0000FF'
        )

        surf = omf.SurfaceElement(
            name='trisurf',
            geometry=omf.SurfaceGeometry(
                vertices=np.random.rand(100, 3),
                triangles=np.floor(np.random.rand(50, 3)*100).astype(int)
            ),
            data=[
                omf.ScalarData(
                    name='rand vert data',
                    array=np.random.rand(100),
                    location='vertices'
                ),
                omf.ScalarData(
                    name='rand face data',
                    array=np.random.rand(50),
                    location='faces'
                )
            ],
            color=[100, 200, 200]
        )

        grid = omf.SurfaceElement(
            name='gridsurf',
            geometry=omf.SurfaceGridGeometry(
                tensor_u=np.ones(10).astype(float),
                tensor_v=np.ones(15).astype(float),
                origin=[50., 50., 50.],
                axis_u=[1., 0, 0],
                axis_v=[0, 0, 1.],
                offset_w=np.random.rand(11, 16).flatten()
            ),
            data=[
                omf.ScalarData(
                    name='rand vert data',
                    array=np.random.rand(11, 16).flatten(),
                    location='vertices'
                ),
                omf.ScalarData(
                    name='rand face data',
                    array=np.random.rand(10, 15).flatten(order='f'),
                    location='faces'
                )
            ],
            textures=[
                omf.ImageTexture(
                    name='test image',
                    image=pngfile,
                    origin=[2., 2., 2.],
                    axis_u=[5., 0, 0],
                    axis_v=[0, 2., 5.]
                )
            ]
        )

        vol = omf.VolumeElement(
            name='vol',
            geometry=omf.VolumeGridGeometry(
                tensor_u=np.ones(10).astype(float),
                tensor_v=np.ones(15).astype(float),
                tensor_w=np.ones(20).astype(float),
                origin=[10., 10., -10]
            ),
            data=[
                omf.ScalarData(
                    name='Random Data',
                    location='cells',
                    array=np.random.rand(10, 15, 20).flatten()
                )
            ]
        )

        proj.elements = [pts, lin, surf, grid, vol]

        assert proj.validate()

        serial_file = os.path.sep.join([dirname, 'out.omf'])
        omf.OMFWriter(proj, serial_file)
        new_proj = omf.OMFReader(serial_file)

        assert new_proj.validate()
        assert str(new_proj.elements[3].textures[0].uid) == \
            str(proj.elements[3].textures[0].uid)

        os.remove(serial_file)


if __name__ == '__main__':
    unittest.main()
