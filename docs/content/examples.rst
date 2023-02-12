.. _examples:

OMF API Example
===============

This (very impractical) example shows usage of the OMF API.

Also, this example builds elements all at once. They can also be initialized
with no arguments, and properties can be set one-by-one (see code snippet at
bottom of page).

.. code:: python
    :name: test_doc

    import datetime
    import numpy as np
    import os
    import png
    import omf

    # setup sample files
    dir = os.getcwd()
    png_file = os.path.join(dir, "example.png")
    omf_file = os.path.join(dir, "example.omf")
    for f in (png_file, omf_file):
        if os.path.exists(f):
            os.remove(f)
    img = ["110010010011", "101011010100", "110010110101", "100010010011"]
    img = [[int(val) for val in value] for value in img]
    writer = png.Writer(len(img[0]), len(img), greyscale=True, bitdepth=16)
    with open(png_file, "wb") as file:
        writer.write(file, img)

    proj = omf.Project(
        name="Test project",
        description="Just some assorted elements",
    )
    pts = omf.PointSet(
        name="Random Points",
        description="Just random points",
        vertices=np.random.rand(100, 3),
        attributes=[
            omf.NumericAttribute(
                name="rand attr",
                array=np.random.rand(100),
                location="vertices",
            ),
            omf.NumericAttribute(
                name="More rand attr",
                array=np.random.rand(100),
                location="vertices",
            ),
        ],
        textures=[
            omf.ProjectedTexture(
                name="test image",
                image=png_file,
                origin=[0, 0, 0],
                axis_u=[1, 0, 0],
                axis_v=[0, 1, 0],
            ),
            omf.ProjectedTexture(
                name="test image",
                image=png_file,
                origin=[0, 0, 0],
                axis_u=[1, 0, 0],
                axis_v=[0, 0, 1],
            ),
        ],
        metadata={
            "color": "green",
        },
    )
    lin = omf.LineSet(
        name="Random Line",
        vertices=np.random.rand(100, 3),
        segments=np.floor(np.random.rand(50, 2) * 100).astype(int),
        attributes=[
            omf.NumericAttribute(
                name="rand vert attr",
                array=np.random.rand(100),
                location="vertices",
            ),
            omf.NumericAttribute(
                name="rand segment attr",
                array=np.random.rand(50),
                location="segments",
            ),
        ],
        metadata={
            "color": "#0000FF",
        },
    )
    surf = omf.Surface(
        name="trisurf",
        vertices=np.random.rand(100, 3),
        triangles=np.floor(np.random.rand(50, 3) * 100).astype(int),
        attributes=[
            omf.NumericAttribute(
                name="rand vert attr",
                array=np.random.rand(100),
                location="vertices",
            ),
            omf.NumericAttribute(
                name="rand face attr",
                array=np.random.rand(50),
                location="faces",
            ),
        ],
        metadata={
            "color": [100, 200, 200],
        },
    )
    grid = omf.TensorGridSurface(
        name="gridsurf",
        tensor_u=np.ones(10).astype(float),
        tensor_v=np.ones(15).astype(float),
        origin=[50.0, 50.0, 50.0],
        axis_u=[1.0, 0, 0],
        axis_v=[0, 0, 1.0],
        offset_w=np.random.rand(11 * 16),
        attributes=[
            omf.NumericAttribute(
                name="rand vert attr",
                array=np.random.rand(11 * 16),
                location="vertices",
            ),
            omf.NumericAttribute(
                name="rand face attr",
                array=np.random.rand(10 * 15),
                location="faces",
            ),
        ],
        textures=[
            omf.ProjectedTexture(
                name="test image",
                image=png_file,
                origin=[2.0, 2.0, 2.0],
                axis_u=[5.0, 0, 0],
                axis_v=[0, 2.0, 5.0],
            ),
        ],
    )
    vol = omf.TensorGridBlockModel(
        name="vol",
        tensor_u=np.ones(10).astype(float),
        tensor_v=np.ones(15).astype(float),
        tensor_w=np.ones(20).astype(float),
        origin=[10.0, 10.0, -10],
        attributes=[
            omf.NumericAttribute(
                name="random attr", location="cells", array=np.random.rand(10 * 15 * 20)
            ),
        ],
    )

    proj.elements = [pts, lin, surf, grid, vol]

    proj.metadata = {
        "coordinate_reference_system": "epsg 3857",
        "date_created": datetime.datetime.utcnow(),
        "version": "v1.3",
        "revision": "10",
    }

    assert proj.validate()

    omf.save(proj, omf_file)


Piecewise building example:

.. code:: python

    ...
    pts = omf.PointSet()
    pts.name = 'Random Points',
    pts.vertices = np.random.rand(100, 3)
    ...
