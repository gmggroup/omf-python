.. _data:

Data
****

ProjectElements include a list of ProjectElementData. These specify mesh location
('vertices', 'faces', etc.) as well as the array, name, and
description. See class descriptions below for specific types of Data.

Mapping array values to a mesh is staightforward for unstructured meshes
(those defined by vertices, segments, triangles, etc); the order of the data
array simply corresponds to the order of the associated mesh parameter.
For grid meshes, however, mapping 1D data array to the 2D or 3D grid requires
correctly ordered unwrapping. The default is C-style, row-major ordering,
:code:`order='c'`. To align data this way, you may start with a numpy array
that is size (x, y) for 2D data or size (x, y, z) for 3D data then use
numpy's :code:`flatten()` function with default order 'C'. Alternatively,
if your data uses Fortran- or Matlab-style, column-major ordering, you may
specify data :code:`order='f'`.

Here is a code snippet to show data binding in action; this assumes
the surface contains a mesh with 9 vertices and 4 faces (ie a 2x2 square grid).


.. code:: python

    ...
    my_surface = omf.Surface(...)
    ...
    my_node_data = omf.ScalarData(
        name='Nine Numbers',
        array=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        location='vertices',
        order='c'  # Default
    )
    my_face_data = omf.ScalarData(
        name='Four Numbers',
        array=[0.0, 1.0, 2.0, 3.0],
        location='faces'
    )
    my_surface.data = [
        my_face_data,
        my_node_data
    ]

ScalarData
----------

.. autoclass:: omf.data.ScalarData

Vector3Data
-----------

.. autoclass:: omf.data.Vector3Data

Vector2Data
-----------

.. autoclass:: omf.data.Vector2Data

ColorData
---------

.. autoclass:: omf.data.ColorData

StringData
----------

.. autoclass:: omf.data.StringData

DateTimeData
------------

.. autoclass:: omf.data.DateTimeData

MappedData
----------

.. autoclass:: omf.data.MappedData

Legend
------

.. autoclass:: omf.data.Legend

ScalarColormap
--------------

.. autoclass:: omf.data.ScalarColormap

DateTimeColormap
----------------

.. autoclass:: omf.data.DateTimeColormap
