.. _surfaces:

Surface
*******

Transfer geological domains from 3D modelling software to Resource Estimation software.

.. image:: /images/Surface.png

Element
-------

.. autoclass:: omf.surface.SurfaceElement

Geometry
--------

Surfaces have two available geometries: SurfaceGeometry, an unstructured triangular mesh,
and SurfaceGridGeometry, a gridded mesh.

SurfaceGeometry
===============

.. image:: /images/SurfaceGeometry.png
    :align: center

.. autoclass:: omf.surface.SurfaceGeometry

SurfaceGridGeometry
===================

.. image:: /images/SurfaceGridGeometry.png
    :align: center

.. autoclass:: omf.surface.SurfaceGridGeometry

Data
----

Data is a list of :ref:`data <data>`. For Surfaces, :code:`location='vertices'`
and :code:`location='faces'` are valid.

Textures
--------

Textures are :ref:`ImageTexture <textures>` mapped to the Surface.
