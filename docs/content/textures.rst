.. _textures:

Texture
*******

Textures are images that exist in space and are mapped to their corresponding
elements. Unlike data, they do not need to correspond to mesh nodes or
cell centers. This image shows how textures are mapped to a surface. Their
position is defined by an origin, O, and axis vectors, U and V, then they
are mapped laterally to the element position.

.. image:: /images/ImageTexture.png

Like data, multiple textures can be applied to a element. Simply provide a
list of textures.

.. code:: python

    ...
    my_surface = omf.SurfaceElement(...)
    ...
    my_tex_1 = omf.ImageTexture(
        O=[0.0, 0.0, 0.0],
        U=[1.0, 0.0, 0.0],
        V=[0.0, 1.0, 0.0],
        image='image1.png'
    )
    my_tex_2 = omf.ImageTexture(
        O=[0.0, 0.0, 0.0],
        U=[1.0, 0.0, 0.0],
        V=[0.0, 0.0, 1.0],
        image='image2.png'
    )
    my_surface.textures = [
        my_tex_1,
        my_tex_2
    ]

.. autoclass:: omf.texture.ImageTexture
