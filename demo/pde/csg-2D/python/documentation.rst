.. Documentation for the csg 2D demo from DOLFIN.

.. _demo_pde_csg-2D_python_documentation:


Create CSG geometry
===================

This demo is implemented in a single Python file,
:download:`demo_csg_2D.py`, and demonstrates use of 2D geometries in DOLFIN.

.. include:: ../common.txt


Implementation
--------------

This description goes through how to make geometries and meshes in DOLFIN.

First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

	from dolfin import *

Then we check if CGAL is installed, as it is needed to compile this demo:

.. code-block:: python

	if not has_cgal():
    		print "DOLFIN must be compiled with CGAL to run this demo."
    		exit(0)

Now, we define 2D geometries. 
We create a :py:class:`Rectangle <dolfin.cpp.mesh.Rectangle>` defined by two opposite corners:

.. code-block:: python

	r = Rectangle(0.5, 0.5, 1.5, 1.5)

where the first two arguments represents the first corner, and the last two arguments represents the opposite corner. 
A :py:class:`Circle <dolfin.cpp.mesh.Circle>`  may be defined by:

.. code-block:: python

	c = Circle (1, 1, 1)

where the center of the circle is given by the first two arguments, and the third argument gives the radius. 
We may use these geometries to define a new geometry by subtracting one from the other:

.. code-block:: python

	g2d = c - r

To get information about our new geometry we use the function :py:func:`info <dolfin.cpp.common.info>`. 
This function takes a string or a DOLFIN object as argument, and optionally we can give a second argument 
to indicate whether verbose object data should be printed. 
If the second argument is False (which is default), a one-line summary is printed. 
If True, verbose and sometimes very exhaustive object data are printed.

.. code-block:: python

	# Test printing
	info("\nCompact output of 2D geometry:")
	info(g2d)
	info("")
	info("\nVerbose output of 2D geometry:")
	info(g2d, True)

To visualize our geometry we :py:func:`plot <dolfin.common.plotting.plot>` it:

.. code-block:: python

	# Plot geometry
	plot(g2d, "2D Geometry (boundary)")

The second argument is optional, it specifies title of the plot.

Finally, we generate a mesh using :py:class:`Mesh <dolfin.cpp.mesh.Mesh>` and plot it.

.. code-block:: python

	# Generate and plot mesh
	mesh2d = Mesh(g2d, 10)
	plot(mesh2d, title="2D mesh")

Note that when we create a mesh from a CSG geometry, the resolution must be specified. 
It is given by an integer as a second argument in Mesh.

Complete code
-------------

.. literalinclude:: demo_csg_2D.py
   :start-after: # Begin demo


