.. Documentation for the mesh generation demo from DOLFIN.

.. _demo_pde_mesh-generation_python_documentation:



Generate mesh
=============

This demo is implemented in a Python file,
:download:`demo_mesh_generaton.py`, and the 3D geometries are described in two Object File Format (.off) files, :download:`../tetrahedron.off`, :download:`../cube.off`. 

.. include:: ../common.txt

Implementation
--------------

This description goes through the implementation (in
:download:`demo_mesh_generaton.py`).

First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

    from dolfin import *

:py:class:`PolygonalMeshGenerator <dolfin.cpp.mesh.PolygonalMeshGenerator>` and :py:class:`PolyhedralMeshGenerator <dolfin.cpp.mesh.PolyhedralMeshGenerator>` need CGAL to generate the meshes, so we check that dolfin is compiled with CGAL.

.. code-block:: python

	if not has_cgal():
	    print "DOLFIN must be compiled with CGAL to run this demo."
	    exit(0)

We need an empty mesh to send as argument to the MeshGenerators. 

.. code-block:: python

	# Create empty Mesh
	mesh = Mesh()

Now, we are able to start making the geometries. We start with a polygon, and define it by listing all the points we want the boundary to follow. We need a closed contour, so the last point is positioned at the same place as the first. We represent the points with instances of :py:class:`Point <dolfin.cpp.mesh.Point>`. Since we want an 2D geometry we use the default value 0 for z.  

.. code-block:: python

	# Create list of polygonal domain vertices
	domain_vertices = [Point(0.0, 0.0),
		           Point(10.0, 0.0),
		           Point(10.0, 2.0),
		           Point(8.0, 2.0),
		           Point(7.5, 1.0),
		           Point(2.5, 1.0),
		           Point(2.0, 4.0),
		           Point(0.0, 4.0),
		           Point(0.0, 0.0)]

We send our list of points to :py:class:`PolygonalMeshGenerator <dolfin.cpp.mesh.PolygonalMeshGenerator>` along with the empty mesh and the resolution given by the third argument cell size. We set interactive to True so that we are able to rotate, resize and translate the mesh. 

.. code-block:: python

	# Generate mesh and plot
	PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.25);
	plot(mesh, interactive=True)

.. image:: plot_polygonmesh.png
	:scale: 75 %


The geometry for the next two meshes are described by .off-files (:download:`../tetrahedron.off`, :download:`../cube.off`). One can easily make own .off-file. It consist of four parts:

* The first line is just OFF
* The second line consist of three numbers: the first being number of vertices, the second being number og faces and the will not be used so we set it to 0. 
* From line three we list the vertices, one coordinat (three numbers) on each line. 
* The last part of the file describes the faces (facets, sides) of the geometry. One face is described on one line where the first number says how many vertices we need to represent the face (in a cube we need four, but in a tetrahedral we need three). We then list the vertices describing the face, we use the vertices defined above and start our "indexing" at 0. 

We send the empty mesh, the off-file anf the resolution (cell size) to :py:class:`PolyhedralMeshGenerator <dolfin.cpp.mesh.PolyhedralMeshGenerator>` 

.. code-block:: python

	# Generate 3D mesh from OFF file input (tetrahedron)
	PolyhedralMeshGenerator.generate(mesh, "../tetrahedron.off", 0.05)
	plot(mesh, interactive=True)

	# Generate 3D mesh from OFF file input (cube)
	PolyhedralMeshGenerator.generate(mesh, "../cube.off", 0.05)
	plot(mesh, interactive=True)

.. image:: plot_tetrahedralmesh.png
	:scale: 75 %

.. image:: plot_cubemesh.png
	:scale: 75 %

Complete code
-------------

.. literalinclude:: demo_mesh_generaton.py
   :start-after: # Begin demo
