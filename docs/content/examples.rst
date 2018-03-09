.. _examples:

OMF API Example
===============

This (very impractical) example shows usage of the OMF API.

Also, this example builds elements all at once. They can also be initialized
with no arguments, and properties can be set one-by-one (see code snippet at
bottom of page).

.. code:: python

    import numpy as np
    import omf

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
                image='test_image.png',
                origin=[0, 0, 0],
                axis_u=[1, 0, 0],
                axis_v=[0, 1, 0]
            ),
            omf.ImageTexture(
                name='test image',
                image='test_image.png',
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
                image='test_image.png',
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

    omf.OMFWriter(proj, 'omfproj.omf')


Piecewise building example:

.. code:: python

    ...
    pts = omf.PointSetElement()
    pts.name = 'Random Points',
    pts.mesh = omf.PointSetGeometry()
    pts.mesh.vertices = np.random.rand(100, 3)
    ...
