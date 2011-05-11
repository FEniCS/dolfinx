"""This module provides a dictionary of code examples using the Python syntax.

The idea is to substitute the code examples from the *.h files, which uses the
C++ syntax, with code snippets from this dictionary."""

# Copyright (C) 2010 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2010-10-14
# Last changed: 2010-10-19

codesnippets = {
"Mesh":{
"uint num_cells() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.num_cells()
    8
""",
"uint num_vertices() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.num_vertices()
    9
""",
"uint num_edges() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.num_edges()
    0
    >>> mesh.init(1)
    16
    >>> mesh.num_edges()
    16
""",
"uint num_faces() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.num_faces()
    8
""",
"uint num_facets() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.num_facets()
    0
    >>> mesh.init(0,1)
    >>> mesh.num_facets()
    16
""",
"uint num_entities(uint d) const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.init(0,1)
    >>> mesh.num_entities(0)
    9
    >>> mesh.num_entities(1)
    16
    >>> mesh.num_entities(2)
    8
""",
"double* coordinates()":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(1,1)
    >>> mesh.coordinates()
    array([[ 0.,  0.],
           [ 1.,  0.],
           [ 0.,  1.],
           [ 1.,  1.]])
""",
"const uint* cells() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(1,1)
    >>> mesh.cells()
    array([[0, 1, 3],
          [0, 2, 3]])
""",
"uint size(uint dim) const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.init(0,1)
    >>> mesh.size(0)
    9
    >>> mesh.size(1)
    16
    >>> mesh.size(2)
    8
""",
"dolfin::uint closest_cell(const Point& point) const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(1, 1)
    >>> point = dolfin.Point(0.0, 2.0)
    >>> mesh.closest_cell(point)
    1
""",
"double hmin() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.hmin()
    0.70710678118654757
""",
"double hmax() const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.hmax()
    0.70710678118654757
""",
"std::string str(bool verbose) const":
"""
.. code-block:: python

    >>> mesh = dolfin.UnitSquare(2,2)
    >>> mesh.str(False)
    '<Mesh of topological dimension 2 (triangles) with 9 vertices and 8 cells, ordered>'
"""
},
"MeshEntityIterator":
{
"MeshEntityIterator":
"""
The following example shows how to iterate over all mesh entities
of a mesh of topological dimension dim:

.. code-block:: python

    >>> for e in dolfin.cpp.entities(mesh, 1):
    ...     print e.index()

The following example shows how to iterate over mesh entities of
topological dimension dim connected (incident) to some mesh entity f:

.. code-block:: python

    >>> f = dolfin.cpp.MeshEntity(mesh, 0, 0)
    >>> for e in dolfin.cpp.entities(f, 1):
    ...     print e.index()
"""
},
"Event":
{
"Event":
"""
.. code-block:: python

    >>> event = dolfin.Event("System is stiff, damping is needed.", 3)
    >>> for i in range(10):
    ...     if i > 7:
    ...         print i
    ...         event()
"""
},
"Progress":
{
"Progress":
"""
A progress bar may be used either in an iteration with a known number
of steps:

.. code-block:: python

    >>> n = 1000000
    >>> p = dolfin.Progress("Iterating...", n)
    >>> for i in range(n):
    ...     p += 1

or in an iteration with an unknown number of steps:

.. code-block:: python

    >>> pr = dolfin.Progress("Iterating")
    >>> t = 0.0
    >>> n = 1000000.0
    >>> while t < n:
    ...     t += 1.0
    ...     p += t/n
"""
}
}

