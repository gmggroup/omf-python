.. _textures:

Projected Texture
*****************

Projected textures are images that exist in space and are mapped to their
corresponding elements. Unlike data, they do not need to correspond to mesh
nodes or cell centers. This image shows how textures are mapped to a surface.
Their position is defined by an origin and axis vectors then they
are mapped laterally to the element position.

.. image:: /docs/images/ImageTexture.png

Like data, multiple textures can be applied to a element; simply provide a
list of textures. Each of these textures provides an origin point and two
extent vectors for the plane defining where images rests.
The `axis_*` properties define the extent of that image out from the origin.
Given a rectangular PNG image, the `origin` is the bottom left,
`origin + axis_u` is the bottom right, and `origin + axis_v` is the top left.
This allows the image to be rotated and/or skewed.
These values are independent of the corresponding Surface; in fact, there is
nothing requiring the image to actually align with the Surface.

.. code:: python

    >> ...
    >> my_surface = omf.SurfaceElement(...)
    >> ...
    >> my_tex_1 = omf.ProjectedTexture(
           origin=[0.0, 0.0, 0.0],
           axis_u=[1.0, 0.0, 0.0],
           axis_v=[0.0, 1.0, 0.0],
           image='image1.png'
       )
    >> my_tex_2 = omf.ProjectedTexture(
           origin=[0.0, 0.0, 0.0],
           axis_u=[1.0, 0.0, 0.0],
           axis_v=[0.0, 0.0, 1.0],
           image='image2.png'
       )
    >> my_surface.textures = [
           my_tex_1,
           my_tex_2
       ]

.. autoclass:: omf.texture.ProjectedTexture


UV Mapped Textures
******************

Rather than being projected onto points or a surface, UV Mapped Textures
are given normalized UV coordinates which correspond to element
vertices. This allows arbitrary mapping of images to surfaces.

.. autoclass:: omf.texture.UVMappedTexture
