.. Documentation for the biharmonic demo from DOLFIN.

.. _demos_pde_biharmonic_cpp_documentation:


Biharmonic equation
===================

.. include:: ../common.txt


Implementation
--------------

The implementation is split in two files, a form file containing the definition
of the variational forms expressed in UFL and the solver which is implemented
in a C++ file.

Running this demo requires the files: :download:`main.cpp`,
:download:`Biharmonic.ufl` and :download:`CMakeLists.txt`.

UFL form file
^^^^^^^^^^^^^

First we define the variational problem in UFL in the file called
:download:`Biharmonic.ufl`.

In the UFL file, the finite element space is defined:

.. code-block:: python

    # Elements
    element = FiniteElement("Lagrange", triangle, 2)

On the space ``element``, trial and test functions, and the source term
are defined:

.. code-block:: python

    # Trial and test functions
    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)

Next, the outward unit normal to cell boundaries and a measure of the
cell size are defined. The average size of cells sharing a facet will
be used (``h_avg``).  The UFL syntax ``('+')`` and ``('-')`` restricts
a function to the ``('+')`` and ``('-')`` sides of a facet,
respectively.  The penalty parameter ``alpha`` is made a
:cpp:class:`Constant` so that it can be changed in the program without
regenerating the code.

.. code-block:: python

    # Normal component, mesh size and right-hand side
    n  = element.cell().n
    h = 2.0*triangle.circumradius
    h_avg = (h('+') + h('-'))/2

    # Parameters
    alpha = Constant(triangle)

Finally the bilinear and linear forms are defined. Integrals over
internal facets are indicated by ``*dS``.

.. code-block:: python

    # Bilinear form
    a = inner(div(grad(u)), div(grad(v)))*dx \
      - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
      - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
      + alpha('+')/h_avg*inner(jump(grad(u), n), jump(grad(v),n))*dS

    # Linear form
    L = f*v*dx


C++ program
^^^^^^^^^^^

The DOLFIN interface and the code generated from the UFL input is included,
and the DOLFIN namespace is used:

.. code-block:: c++

  #include <dolfin.h>
  #include "Biharmonic.h"

  using namespace dolfin;

A class ``Source`` is defined for the function :math:`f`, with the
function ``Expression::eval`` overloaded:

.. code-block:: c++

  // Source term
  class Source : public Expression
  {
  public:

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 4.0*std::pow(DOLFIN_PI, 4)*std::sin(DOLFIN_PI*x[0])*std::sin(DOLFIN_PI*x[1]);
    }

  };

A boundary subdomain is defined, which in this case is the entire boundary:

.. code-block:: c++

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return on_boundary;
    }
  };

The main part of the program is begun, and a mesh is created with 32 vertices
in each direction:

.. code-block:: c++

  int main()
  {
    // Create mesh
    UnitSquare mesh(32, 32);


The source function, a function for the cell size and the penalty term
are declared:

.. code-block:: c++

    // Create functions
    Source f;
    Constant alpha(8.0);

A function space object, which is defined in the generated code, is created:

.. code-block:: c++

    // Create function space
    Biharmonic::FunctionSpace V(mesh);

The Dirichlet boundary condition on :math:`u` is constructed by
defining a :cpp:class:`Constant` which is equal to zero, defining the boundary
(``DirichletBoundary``), and using these, together with ``V``, to create
``bc``:

.. code-block:: c++

    // Define boundary condition
    Constant u0(0.0);
    DirichletBoundary boundary;
    DirichletBC bc(V, u0, boundary);

Using the function space ``V``, the bilinear and linear forms
are created, and function are attached:

.. code-block:: c++

    // Define forms and attach functions
    Biharmonic::BilinearForm a(V, V);
    Biharmonic::LinearForm L(V);
    a.alpha = alpha; L.f = f;

A :cpp:class:`VariationalProblem` is created from the forms and the
Dirichet boundary condition, a finite element function ``u`` is
created and the problem is solved:

.. code-block:: c++

    // Create PDE
    VariationalProblem problem(a, L, bc);

    // Solve PDE
    Function u(V);
    problem.solve(u);

The solution is then plotted to the screen and written to a file in VTK
format:

.. code-block:: c++

    // Plot solution
    plot(u);

    // Save solution in VTK format
    File file("biharmonic.pvd");
    file << u;

    return 0;
  }

Complete code
-------------

Complete UFL file
^^^^^^^^^^^^^^^^^

.. literalinclude:: Biharmonic.ufl
   :start-after: # Compile
   :language: python

Complete main file
^^^^^^^^^^^^^^^^^^

.. literalinclude:: main.cpp
   :start-after: // using
   :language: c++
