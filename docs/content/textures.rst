.. _textures:

Textures
********

Projected Texture
-----------------

Projected textures are images that exist in space and are mapped to their
corresponding elements. Unlike attributes, they do not need to correspond to mesh
nodes or cell centers. This image shows how textures are mapped to a surface.
Their position is defined by a corner and axis vectors then they
are mapped laterally to the element position.

.. image:: /images/ImageTexture.png

Like attributes, multiple textures can be applied to a element; simply provide a
list of textures. Each of these textures provides a corner point and two
extent vectors for the plane defining where images rests.
The `axis_*` properties define the extent of that image out from the corner.
Given a rectangular PNG image, the `corner` is the bottom left,
`corner + axis_u` is the bottom right, and `corner + axis_v` is the top left.
This allows the image to be rotated and/or skewed.
These values are independent of the corresponding Surface; in fact, there is
nothing requiring the image to actually align with the Surface.

.. autoclass:: omf.texture.ProjectedTexture


UV Mapped Textures
------------------

Rather than being projected onto points or a surface, UV Mapped Textures
are given normalized UV coordinates which correspond to element
vertices. This allows arbitrary mapping of images to surfaces.

.. autoclass:: omf.texture.UVMappedTexture

Image
-----

.. autoclass:: omf.texture.Image
