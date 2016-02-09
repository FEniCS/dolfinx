.. Documentation for the DOLFIN iterative Stokes demo.

.. _demo_pde_subdomains_poisson_documentation:

Poisson equation with multiple subdomains
=========================================

This demo is implemented in a single Python file,
:download:`demo_subdomains-poisson.py`, which contains both the
variational forms and the solver. We suggest that you familiarize
yourself with the :ref:`Poisson demo
<demo_pde_poisson_python_documentation>` before studying this example,
as some of the more standard steps will be described in less detail.

.. include:: ../common.txt

Implementation
--------------

This description goes through the implementation (in
:download:`demo_subdomains-poisson.py`) of a solver for the above
described equation.

In this example, different boundary conditions are prescribed on
different parts of the boundaries, and different parts of the interior
have different material properties. This information must be made
available to the solver.  One way of doing this, is to tag the
different subregions with different (integer) labels, and later
integrate over the specified regions. DOLFIN provides a class
:py:class:`MeshFunction <dolfin.cpp.MeshFunction>` which is useful for
these types of operations: instances of this class represent functions
over mesh entities (such as over cells or over facets). Mesh functions
can be read from file or, if explicit formulae for the domains are
known, they can be constructed by way of instances of the
:py:class:`SubDomain <dolfin.cpp.SubDomain>` class. The latter is the
case here, so we begin by defining the left, right, top and bottom
boundaries, and the interior obstacle domain using the
:py:class:`SubDomain <dolfin.cpp.SubDomain>` class and creating
instances of these classes.

.. code-block:: python

    from dolfin import *

    # Create classes for defining parts of the boundaries and the interior
    # of the domain
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0)

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.0)

    class Obstacle(SubDomain):
        def inside(self, x, on_boundary):
            return (between(x[1], (0.5, 0.7)) and between(x[0], (0.2, 1.0)))

    # Initialize sub-domain instances
    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    obstacle = Obstacle()

Note that the DOLFIN functions :py:func:`near <dolfin.cpp.near>` and
:py:func:`between <dolfin.cpp.between>` provide robust ways of testing
whether a coordinate is (to within machine precision) close to a given
numerical value and in a range of values, respectively.

We next define a mesh of the domain:

.. code-block:: python

    mesh = UnitSquareMesh(64, 64)

The above subdomains are defined with the sole purpose of populating
mesh functions. (For more complicated geometries, the mesh functions
would typically be provided by other means.) The classes
:py:class:`CellFunction <dolfin.cpp.CellFunction>` and
:py:class:`FacetFunction <dolfin.cpp.FacetFunction>` are specialized
versions of the more general :py:class:`MeshFunction
<dolfin.cpp.MeshFunction>`. :py:class:`CellFunction
<dolfin.cpp.CellFunction>` represents a function with a value for each
cell of a mesh, while :py:class:`FacetFunction
<dolfin.cpp.FacetFunction>` represents a function with a value for
each facet. We define a :py:class:`CellFunction
<dolfin.cpp.CellFunction>` to indicate which cells that correspond to
the different interior subregions :math:`\Omega_0` and
:math:`\Omega_1`. Those in the interior rectangle will be tagged by
`1`, while the remainder is tagged by `0`. We can set all the values
of a :py:class:`MeshFunction <dolfin.cpp.MeshFunction>` to a given
value using the :py:func:`set_all
<dolfin.cpp.MeshFunctionInt.set_all>` method.  So, in order to
accomplish what we want, we can set all values to `0` first, and then
we can use the ``obstacle`` instance to mark the cells identified as
inside the obstacle region by `1` (thus overwriting the previous
value):

.. code-block:: python

    # Initialize mesh function for interior domains
    domains = CellFunction("size_t", mesh)
    domains.set_all(0)
    obstacle.mark(domains, 1)


We can do the same for the boundaries using a :py:class:`FacetFunction
<dolfin.cpp.FacetFunction>`. We first tag all the edges by ``0``, then
the edges on the left by ``1``, on the top by ``2``, on the right by
``3`` and on the bottom by ``4``:

.. code-block:: python

    # Initialize mesh function for boundary domains
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    top.mark(boundaries, 2)
    right.mark(boundaries, 3)
    bottom.mark(boundaries, 4)

Now that the geometry is defined and labeled, we can move on to
defining the input source functions:

.. code-block:: python

    # Define input data
    a0 = Constant(1.0)
    a1 = Constant(0.01)
    g_L = Expression("- 10*exp(- pow(x[1] - 0.5, 2))", degree=2)
    g_R = Constant(1.0)
    f = Constant(1.0)

Here, ``a0`` and ``a1`` represent the values of the coefficient
:math:`a` in the two regions of the domain, ``g_L`` and ``g_R``
represent the values of the Neumann boundary condition on the left and
right boundaries respectively, and ``f`` represents the body source.

We may now move on to define the variational equation. As usual, we
start by defining a finite element function space and basis functions
on this space:

.. code-block:: python

    # Define function space and basis functions
    V = FunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

With this function space, we can define the essential (Dirichlet)
boundary conditions on the top and bottom boundaries. These boundaries
correspond to the facets tagged by ``2`` and ``4``, respectively, in
the ``boundaries`` facet function:

.. code-block:: python

    # Define Dirichlet boundary conditions at top and bottom boundaries
    bcs = [DirichletBC(V, 5.0, boundaries, 2),
           DirichletBC(V, 0.0, boundaries, 4)]

DOLFIN predefines the "measures" ``dx``, ``ds`` and ``dS``
representing integration over cells, exterior facets (that is, facets
on the boundary) and interior facets, respectively. These measures can
take an additional integer argument.  In fact, ``dx`` defaults to
``dx(0)``, ``ds`` defaults to ``ds(0)``, and ``dS`` defaults to
``dS(0)``. Integration over subregions can be specified by measures
with different integer labels as arguments. However, we also need to
map the geometry information stored in the mesh functions to these
measures. The easiest way of accomplishing this is to define new
measures with the mesh functions as additional input:

.. code-block:: python

    # Define new measures associated with the interior domains and
    # exterior boundaries
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

We can now define the variational forms corresponding to the
variational problem above using these measures and the tags for the
different subregions. For simplicity, we define the full form first,
and then extract the left- and right-hand sides using the UFL
functions :py:func:`lhs` and :py:func:`rhs` afterwards. We can then
:py:func:`solve <dolfin.fem.solving.solve>` as usual:

.. code-block:: python

    # Define variational form
    F = (inner(a0*grad(u), grad(v))*dx(0) + inner(a1*grad(u), grad(v))*dx(1)
         - g_L*v*ds(1) - g_R*v*ds(3)
         - f*v*dx(0) - f*v*dx(1))

    # Separate left and right hand sides of equation
    a, L = lhs(F), rhs(F)

    # Solve problem
    u = Function(V)
    solve(a == L, u, bcs)

Now, we can also evaluate various integrals of the solution or derived
quantities of the solution over different regions, here are some
examples:

.. code-block:: python

    # Evaluate integral of normal gradient over top boundary
    n = FacetNormal(mesh)
    m1 = dot(grad(u), n)*ds(2)
    v1 = assemble(m1)
    print("\int grad(u) * n ds(2) = ", v1)

    # Evaluate integral of u over the obstacle
    m2 = u*dx(1)
    v2 = assemble(m2)
    print("\int u dx(1) = ", v2)

We also plot the solution and its gradient:

.. code-block:: python

    # Plot solution and gradient
    plot(u, title="u")
    plot(grad(u), title="Projected grad(u)")
    interactive()


Complete code
-------------

.. literalinclude:: demo_subdomains-poisson.py
   :start-after: # Begin demo
