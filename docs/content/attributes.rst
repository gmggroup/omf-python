.. _attributes:

Attributes
**********

ProjectElements include a list of ProjectElementAttribute. These specify mesh location
('vertices', 'faces', etc.) as well as the array, name, and
description. See class descriptions below for specific types of Attributes.

Mapping array values to a mesh is staightforward for unstructured meshes
(those defined by vertices, segments, triangles, etc); the order of the attribute
array simply corresponds to the order of the associated mesh parameter.
For grid meshes, however, mapping 1D attribute array to the 2D or 3D grid requires
correctly ordered unwrapping. The default is C-style, row-major ordering,
:code:`order='c'`. To align attributes this way, you may start with a numpy array
that is size (x, y) for 2D attribute or size (x, y, z) for 3D attribute then use
numpy's :code:`flatten()` function with default order 'C'.

Here is a code snippet to show attribute binding in action; this assumes
the surface contains a mesh with 9 vertices and 4 faces (ie a 2x2 square grid).


.. code:: python

    >> ...
    >> my_surface = omf.Surface(...)
    >> ...
    >> my_node_attr = omf.ScalarAttributes(
           name='Nine Numbers',
           array=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
           location='vertices',
           order='c'  # Default
       )
    >> my_face_attr = omf.ScalarAttributes(
           name='Four Numbers',
           array=[0.0, 1.0, 2.0, 3.0],
           location='faces'
       )
    >> my_surface.attributes = [
           my_face_attr,
           my_node_attr
       ]

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
