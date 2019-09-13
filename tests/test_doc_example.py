"""Test the example in the docs"""
import os

import numpy as np
import png

import omf


def test_doc_ex():
    """Acceptance test of the example from the documentation"""
    dirname, _ = os.path.split(os.path.abspath(__file__))
    pngfile = os.path.sep.join([dirname, 'out.png'])
    img = ['110010010011', '101011010100', '110010110101', '100010010011']
    img = [[int(val) for val in value] for value in img]
    writer = png.Writer(len(img[0]), len(img), greyscale=True, bitdepth=16)
    with open(pngfile, 'wb') as file:
        writer.write(file, img)
    proj = omf.Project(
        name='Test project',
        description='Just some assorted elements',
    )
    pts = omf.PointSetElement(
        name='Random Points',
        description='Just random points',
        vertices=np.random.rand(100, 3),
        data=[
            omf.ScalarData(
                name='rand data',
                array=np.random.rand(100),
                location='vertices',
            ),
            omf.ScalarData(
                name='More rand data',
                array=np.random.rand(100),
                location='vertices',
            ),
        ],
        textures=[
            omf.ProjectedTexture(
                name='test image',
                image=pngfile,
                origin=[0, 0, 0],
                axis_u=[1, 0, 0],
                axis_v=[0, 1, 0],
            ),
            omf.ProjectedTexture(
                name='test image',
                image=pngfile,
                origin=[0, 0, 0],
                axis_u=[1, 0, 0],
                axis_v=[0, 0, 1],
            ),
        ],
        color='green',
    )
    lin = omf.LineSetElement(
        name='Random Line',
        vertices=np.random.rand(100, 3),
        segments=np.floor(np.random.rand(50, 2)*100).astype(int),
        data=[
            omf.ScalarData(
                name='rand vert data',
                array=np.random.rand(100),
                location='vertices',
            ),
            omf.ScalarData(
                name='rand segment data',
                array=np.random.rand(50),
                location='segments',
            ),
        ],
        color='#0000FF',
    )
    surf = omf.SurfaceElement(
        name='trisurf',
        vertices=np.random.rand(100, 3),
        triangles=np.floor(np.random.rand(50, 3)*100).astype(int),
        data=[
            omf.ScalarData(
                name='rand vert data',
                array=np.random.rand(100),
                location='vertices',
            ),
            omf.ScalarData(
                name='rand face data',
                array=np.random.rand(50),
                location='faces',
            ),
        ],
        color=[100, 200, 200],
    )
    grid = omf.SurfaceGridElement(
        name='gridsurf',
        tensor_u=np.ones(10).astype(float),
        tensor_v=np.ones(15).astype(float),
        origin=[50., 50., 50.],
        axis_u=[1., 0, 0],
        axis_v=[0, 0, 1.],
        offset_w=np.random.rand(11*16),
        data=[
            omf.ScalarData(
                name='rand vert data',
                array=np.random.rand(11*16),
                location='vertices',
            ),
            omf.ScalarData(
                name='rand face data',
                array=np.random.rand(10*15),
                location='faces',
            ),
        ],
        textures=[
            omf.ProjectedTexture(
                name='test image',
                image=pngfile,
                origin=[2., 2., 2.],
                axis_u=[5., 0, 0],
                axis_v=[0, 2., 5.],
            ),
        ],
    )
    vol = omf.VolumeGridElement(
        name='vol',
        tensor_u=np.ones(10).astype(float),
        tensor_v=np.ones(15).astype(float),
        tensor_w=np.ones(20).astype(float),
        origin=[10., 10., -10],
        data=[
            omf.ScalarData(
                name='Random Data',
                location='cells',
                array=np.random.rand(10*15*20)
            ),
        ],
    )
    proj.elements = [pts, lin, surf, grid, vol]
    assert proj.validate()
    serialfile = os.path.sep.join([dirname, 'out.omf'])
    omf.OMFWriter(proj, serialfile)
    reader = omf.OMFReader(serialfile)
    reader.get_project_overview()
    new_proj = reader.get_project()
    assert new_proj.validate()
    assert str(new_proj.elements[3].textures[0].uid) == \
        str(proj.elements[3].textures[0].uid)
    del reader
    os.remove(pngfile)
    os.remove(serialfile)
