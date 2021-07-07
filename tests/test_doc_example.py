"""Test the example in the docs"""
import datetime
import os

import numpy as np
import png

import omf


def test_doc_ex():
    """Acceptance test of the example from the documentation"""
    dirname, _ = os.path.split(os.path.abspath(__file__))
    pngfile = os.path.sep.join([dirname, 'out.png'])
    serialfile = os.path.sep.join([dirname, 'out.omf'])
    if os.path.exists(pngfile):
        os.remove(pngfile)
    if os.path.exists(serialfile):
        os.remove(serialfile)
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
        attributes=[
            omf.NumericAttribute(
                name='rand attr',
                array=np.random.rand(100),
                location='vertices',
            ),
            omf.NumericAttribute(
                name='More rand attr',
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
        metadata={
            'color': 'green',
        },
    )
    lin = omf.LineSetElement(
        name='Random Line',
        vertices=np.random.rand(100, 3),
        segments=np.floor(np.random.rand(50, 2)*100).astype(int),
        attributes=[
            omf.NumericAttribute(
                name='rand vert attr',
                array=np.random.rand(100),
                location='vertices',
            ),
            omf.NumericAttribute(
                name='rand segment attr',
                array=np.random.rand(50),
                location='segments',
            ),
        ],
        metadata={
            'color': '#0000FF',
        },
    )
    surf = omf.SurfaceElement(
        name='trisurf',
        vertices=np.random.rand(100, 3),
        triangles=np.floor(np.random.rand(50, 3)*100).astype(int),
        attributes=[
            omf.NumericAttribute(
                name='rand vert attr',
                array=np.random.rand(100),
                location='vertices',
            ),
            omf.NumericAttribute(
                name='rand face attr',
                array=np.random.rand(50),
                location='faces',
            ),
        ],
        metadata={
            'color': [100, 200, 200],
        },
    )
    grid = omf.SurfaceGridElement(
        name='gridsurf',
        tensor_u=np.ones(10).astype(float),
        tensor_v=np.ones(15).astype(float),
        origin=[50., 50., 50.],
        axis_u=[1., 0, 0],
        axis_v=[0, 0, 1.],
        offset_w=np.random.rand(11*16),
        attributes=[
            omf.NumericAttribute(
                name='rand vert attr',
                array=np.random.rand(11*16),
                location='vertices',
            ),
            omf.NumericAttribute(
                name='rand face attr',
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
    vol = omf.TensorBlockModel(
        name='vol',
        tensor_u=np.ones(10).astype(float),
        tensor_v=np.ones(15).astype(float),
        tensor_w=np.ones(20).astype(float),
        corner=[10., 10., -10],
        attributes=[
            omf.NumericAttribute(
                name='random attr',
                location='cells',
                array=np.random.rand(10*15*20)
            ),
        ],
    )
    proj.elements = [pts, lin, surf, grid, vol]
    proj.metadata = {
        'coordinate_reference_system': 'epsg 3857',
        'date_created': datetime.datetime.utcnow(),
        'version': 'v1.3',
        'revision': '10',
    }
    assert proj.validate()
    omf.save(proj, serialfile)
    omf.base.BaseModel._INSTANCES = {}                                          #pylint: disable=protected-access
    omf.load(serialfile, include_binary=False)
    omf.base.BaseModel._INSTANCES = {}                                          #pylint: disable=protected-access
    new_proj = omf.load(serialfile)
    assert new_proj.validate()
    os.remove(pngfile)
    os.remove(serialfile)
