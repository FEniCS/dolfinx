.. Documentation for the stokes stabilized demo from DOLFIN.

.. _demo_pde_stokes-stabilized_python_documentation:

Stokes equations with first order elements
==========================================


.. include:: ../common.txt

Implementation
--------------
First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

	from dolfin import *

In this example, different boundary conditions are prescribed on different parts of the boundaries. This information must be made available to the solver. One way of doing this, is to tag the different sub-regions with different (integer) labels. DOLFIN provides a class :py:class:`MeshFunction <dolfin.cpp.mesh.MeshFunction>` which is useful for these types of operations: instances of this class represents a function over mesh entities (such as over cells or over facets). Mesh and mesh functions can be read from file in the following way:

.. code-block:: python
	
	# Load mesh and subdomains
	mesh = Mesh("dolfin_fine.xml.gz")
	sub_domains = MeshFunction("size_t", mesh, "dolfin_fine_subdomains.xml.gz")

Next, we define a :py:class:`MixedFunctionSpace <dolfin.functions.functionspace.MixedFunctionSpace>` composed of a :py:class:`VectorFunctionSpace <dolfin.functions.functionspace.VectorFunctionSpace>` and a :py:class:`FunctionSpace <dolfin.cpp.function.FunctionSpace>`, both of continuous piecewise linears.

.. code-block:: python

	# Define function spaces
	scalar = FunctionSpace(mesh, "CG", 1)
	vector = VectorFunctionSpace(mesh, "CG", 1)
	system = vector * scalar

Now that we have our mixed function space and marked subdomains defining the boundaries, we create functions for the boundary conditions and define boundary conditions:

.. code-block:: python
	
	# Create functions for boundary conditions
	noslip = Constant((0, 0))
	inflow = Expression(("-sin(x[1]*pi)", "0"))
	zero   = Constant(0)

	# No-slip boundary condition for velocity
	bc0 = DirichletBC(system.sub(0), noslip, sub_domains, 0)

	# Inflow boundary condition for velocity
	bc1 = DirichletBC(system.sub(0), inflow, sub_domains, 1)

	# Boundary condition for pressure at outflow
	bc2 = DirichletBC(system.sub(1), zero, sub_domains, 2)

	# Collect boundary conditions
	bcs = [bc0, bc1, bc2]

Here, we have given four arguments in call of :py:class:`DirichletBC <dolfin.cpp.fem.DirichletBC>`. The first specifies the :py:class:`FunctionSpace <dolfin.cpp.function.FunctionSpace>`. Since we have a :py:class:`MixedFunctionSpace <dolfin.functions.functionspace.MixedFunctionSpace>`, we write system.sub(0) for the :py:class:`VectorFunctionSpace <dolfin.functions.functionspace.VectorFunctionSpace>`, and system.sub(1) for the :py:class:`FunctionSpace <dolfin.cpp.function.FunctionSpace>`. The second argument specifies the value on the Dirichlet boundary. The two last ones specifies the marking of the subdomains; sub_domains contains the subdomain markers and the number given as the last argument is the subdomain index.

The bilinear and linear forms corresponding to the stabilized weak mixed formulation of the Stokes equations are defined as follows:

.. code-block:: python
	
	# Define variational problem
	(v, q) = TestFunctions(system)
	(u, p) = TrialFunctions(system)
	f = Constant((0, 0))
	h = CellSize(mesh)
	beta  = 0.2
	delta = beta*h*h
	a = (inner(grad(v), grad(u)) - div(v)*p + q*div(u) + \
	    delta*inner(grad(q), grad(p)))*dx
	L = inner(v + delta*grad(q), f)*dx

To compute the solution we use the bilinear and linear forms, and the boundary condition, but we also need to create a :py:class:`Function <dolfin.cpp.function.Function>` to store the solution(s). The (full) solution will be stored in w, which we initialize using the MixedFunctionSpace system. The actual computation is performed by calling solve with the arguments a, L and bcs. The separate components u and p of the solution can be extracted by calling the :py:meth:`split <dolfin.functions.function.Function.split>` function. 

.. code-block:: python
	
	# Compute solution
	w = Function(system)
	solve(a == L, w, bcs)
	u, p = w.split()

Finally, we can store to file and plot the solutions.

.. code-block:: python
	
	# Save solution in VTK format
	ufile_pvd = File("velocity.pvd")
	ufile_pvd << u
	pfile_pvd = File("pressure.pvd")
	pfile_pvd << p

	# Plot solution
	plot(u)
	plot(p)
	interactive()

Complete code
-------------

.. literalinclude:: demo_stokes-stabilized.py
	:start-after: # Begin demo
