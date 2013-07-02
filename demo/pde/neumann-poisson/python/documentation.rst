.. Documentation for the Neumann-Poisson demo from DOLFIN.

.. _demo_pde_neumann-poisson_python_documentation:

Poisson equation with pure Neumann boundary conditions
======================================================

.. include:: ../common.txt 
Implementation
--------------


This description goes through the implementation in :download:`demo_neumann-poisson.py` of a solver for the above described Poisson equation step-by-step.


First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

    from dolfin import *

We proceed by defining a mesh of the domain. As the unit square is a very standard domain, we can use a built-in mesh provided by the class :py:class:`UnitSquareMesh <dolfin.cpp.UnitSquareMesh>`. In order to create a mesh consisting of :math:`64\times64` squares with each square divided into two triangles, we do as follows:

.. code-block:: python

	# Create mesh
	mesh = UnitSquareMesh(64, 64)

Next, we need to define the function spaces. We define the two function spaces :math:`V` and :math:`R` separately, before combining these into a mixed function space :math:`W`:

.. code-block:: python

	# Define function spaces and mixed (product) space
	V = FunctionSpace(mesh, "CG", 1)
	R = FunctionSpace(mesh, "R", 0)
	W = V * R

The second argument to :py:class:`FunctionSpace <dolfin.functions.functionspace.FunctionSpace>`  specifies the type of finite element family, while the third argument specifies the polynomial degree. The UFL user manual contains a list of all available finite element families and more details. The * operator creates a mixed (product) space :math:`W` from the two separate spaces :math:`V` and :math:`R`. Hence,

.. math::
	W = \{ (v, d) \ \text{such that} \ v \in V, d \in R \}


Now, we need to specify the trial functions (the unknowns) and the test functions on this space. This can be done using :py:class:`TrialFunctions
<dolfin.functions.function.TrialFunction>` and :py:class:`TestFunctions <dolfin.functions.function.TestFunction>` as follows

.. code-block:: python

	# Define trial and test functions
	(u, c) = TrialFunction(W)
	(v, d) = TestFunctions(W)

In order to define the variational form, it only remains to define the source function :math:`f`. :math:`f` is given by a simple mathematical formula, and can be easily declared using the :py:class:`Expression <dolfin.functions.expression.Expression>` class. Note that the string defining :math:`f` uses C++ syntax since, for efficiency, DOLFIN will generate and compile C++ code for these expressions at run-time.


Since we have natural (Neumann) boundary conditions in this problem, we don´t have to implement boundary conditions. This is because Neumann boundary conditions are default in DOLFIN.

To compute the solution we use the bilinear and linear forms, and the boundary condition, but we also need to create a :py:class:`Function <dolfin.functions.function.Function>` to store the solution(s). The (full) solution will be stored in :math:`w`, which we initialize using the :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` W. The actual computation is performed by calling :py:func:`solve
<dolfin.fem.solving.solve>`. The separate components :math:`u` and :math:`c` of the solution can be extracted by calling the split function. Finally, we plot the solutions to examine the result.

.. code-block:: python

	# Compute solution
	w = Function(W)
	solve(a == L, w)
	(u, c) = w.split()

	# Plot solution
	plot(u, interactive=True)

Complete code
-------------

.. literalinclude:: demo_neumann-poisson.py
   :start-after: # Begin demo

