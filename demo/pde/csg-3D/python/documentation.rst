.. Documentation for the csg 3D demo from DOLFIN.

.. _demo_pde_csg_3D_python_documentation:

Create CSG-3D geometry
======================

This demo is implemented in a single Python file, :download:`demo_csg-3D.py`, and demonstration of usage of 3D geometries in DOLFIN

.. include:: ../common.txt

Implementation
--------------

This description goes through how to make 3 dimentional geometries and meshes in DOLFIN.

First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

	from dolfin import *

Then we check if CGAL is installed, as it is needed to compile this demo:

.. code-block:: python

	if not has_cgal():
		print "DOLFIN must be compiled with CGAL to run this demo."
		exit(0)

Now we define 3D geometries. We start with defining a box by sending the coordinates of two opposite corners as arguments to the class in dolfin called :py:class:`Box <dolfin.cpp.mesh.Box>`. 

.. code-block:: python

	box = Box(0, 0, 0, 1, 1, 1)

We then use :py:class:`Sphere <dolfin.cpp.mesh.Sphere>` to define a sphere with center at :py:class:`Point <dolfin.cpp.mesh.Point>` (:math:`x,y,z`) and radius given with the second argument

.. code-block:: python
	
	sphere = Sphere(Point(0, 0, 0), 0.3) 

To define a :py:class:`Cone <dolfin.cpp.mesh.Cone>` by four arguments, the first being the center at one end :py:class:`Point <dolfin.cpp.mesh.Point>`(:math:`x_1,y_1,z_1`) and the second being the center at the other end :py:class:`Point <dolfin.cpp.mesh.Point>` (:math:`x_2,y_2,z_2`). The two last arguments gives the radius at the ends. 

.. code-block:: python 
 
	cone = Cone(Point(0, 0, -1), Point(0, 0, 1), 1., .5)

Now we have some geometries that we can play with, and the following might seem like magic, but trust me it works!

.. code-block:: python 

	g3d = box + cone - sphere

This simple line makes a geometry of the box and cone merged, but where we take away the area of the sphere.

To get information about our new geometry we use the function :py:func:`info <dolfin.cpp.common.info>`. This function takes a string or a DOLFIN object as argument, and optionally we can give a second argument to indicate whether verbose object data should be printed. If the second argument is False (which is default), a one-line summary is printed. If True, verbose and sometimes very exhaustive object data are printed.

.. code-block:: python

	# Test printing
	info("\nCompact output of 3D geometry:")
	info(g3d)
	info("\nVerbose output of 3D geometry:")
	info(g3d, True)

To visualize our geometry we :py:meth:`plot <dolfin.cpp.io.VTKPlotter.plot>` it:

.. code-block:: python

	plot(g3d, "3D geometry (surface)")

The second argument is optional, it specifies title of the plot.

Finally, we generate a mesh using :py:class:`Mesh <dolfin.cpp.mesh.Mesh>` and plot it.

.. code-block:: python 

	mesh3d = Mesh(g3d, 32)
	info(mesh3d)
	plot(mesh3d, "3D mesh")

Note that when we create a mesh from a CSG geometry, the resolution must be specified.


Complete code
-------------

.. literalinclude:: demo_csg-3D.py
   :start-after: # Begin demo
