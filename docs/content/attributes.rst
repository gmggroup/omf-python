.. _attributes:

Attributes
**********

ProjectElements include a list of ProjectElementAttribute. These specify mesh location
('vertices', 'faces', etc.) as well as the array, name, and
description. See class descriptions below for specific types of Attributes.

Mapping attribute array values to a mesh is straightforward for unstructured meshes
(those defined by vertices, segments, triangles, etc); the order of the attribute
array corresponds to the order of the associated mesh parameter.
For grid meshes, however, mapping 1D attribute array to the 2D or 3D grid requires
correctly ordered ijk unwrapping.

NumericAttribute
----------------

.. autoclass:: omf.attribute.NumericAttribute

VectorAttribute
---------------

.. autoclass:: omf.attribute.VectorAttribute

StringAttribute
---------------

.. autoclass:: omf.attribute.StringAttribute

CategoryAttribute
-----------------

.. autoclass:: omf.attribute.CategoryAttribute

ContinuousColormap
------------------

.. autoclass:: omf.attribute.ContinuousColormap

DiscreteColormap
----------------

.. autoclass:: omf.attribute.DiscreteColormap

CategoryColormap
----------------

.. autoclass:: omf.attribute.CategoryColormap
