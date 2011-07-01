.. Documentation for the hyperelasticity demo from DOLFIN.

.. _demo_pde_hyperelasticity_cpp_documentation:

Hyperelasticity
===============

.. include:: ../common.txt

Implementation
--------------

The implementation is split in two files: a form file containing the
definition of the variational forms expressed in UFL and the solver which is
implemented in a C++ file.

Running this demo requires the files: :download:`main.cpp`,
:download:`HyperElasticity.ufl` and :download:`CMakeLists.txt`.

UFL form file
^^^^^^^^^^^^^

The first step is to define the variational problem at hand. We define
the variational problem in UFL terms in a separate form file
:download:`HyperElasticity.ufl`.

We are interested in solving for a discrete vector field in three
dimensions, so first we need the appropriate finite element space and
trial and test functions on this space

.. literalinclude:: HyperElasticity.ufl
  :lines: 12-17

Note that ``VectorElement`` creates a finite element space of vector
fields. The dimension of the vector field (the number of components)
is assumed to be the same as the spatial dimension (in this case 3),
unless otherwise specified.

Next, we will be needing functions for the boundary source ``B``, the
traction ``T`` and the displacement solution itself ``u``.

.. literalinclude:: HyperElasticity.ufl
  :lines: 19-22

Now, we can define the kinematic quantities involved in the model

.. literalinclude:: HyperElasticity.ufl
  :lines: 24-31

Before defining the energy density and thus the total potential
energy, it only remains to specify constants for the elasticity
parameters

.. literalinclude:: HyperElasticity.ufl
  :lines: 33-41

Both the first variation of the potential energy, and the Jacobian of
the variation, can be automatically computed by a call to
``derivative``:

.. literalinclude:: HyperElasticity.ufl
  :lines: 43-47

Note that ``derivative`` is here used with three arguments: the form
to be differentiated, the variable (function) we are supposed to
differentiate with respect too, and the direction the derivative is
taken in.

Before the form file can be used in the C++ program, it must be
compiled using FFC by running (on the command-line):

.. code-block:: sh

    ffc -l dolfin HyperElasticity.ufl

Note the flag ``-l dolfin`` which tells FFC to generate
DOLFIN-specific wrappers that make it easy to access the generated
code from within DOLFIN.

C++ program
^^^^^^^^^^^

The main solver is implemented in the :download:`main.cpp` file.

At the top, we include the DOLFIN header file and the generated header
file "HyperElasticity.h" containing the variational forms and function
spaces.  For convenience we also include the DOLFIN namespace.

.. code-block:: c++

  #include <dolfin.h>
  #include "HyperElasticity.h"

  using namespace dolfin;

We begin by defining two classes, deriving from ``SubDomain`` for
later use when specifying domains for the boundary conditions.

.. code-block:: c++

  // Sub domain for clamp at left end
  class Left : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return (std::abs(x[0]) < DOLFIN_EPS) && on_boundary;
    }
  };

  // Sub domain for rotation at right end
  class Right : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return (std::abs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary;
    }
  };

We also define two classes, deriving from ``Expression``, for later
use when specifying values for the boundary conditions.

.. code-block:: c++

  // Dirichlet boundary condition for clamp at left end
  class Clamp : public Expression
  {
  public:

    Clamp() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }

  };

  // Dirichlet boundary condition for rotation at right end
  class Rotation : public Expression
  {
  public:

    Rotation() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      const double scale = 0.5;

      // Center of rotation
      const double y0 = 0.5;
      const double z0 = 0.5;

      // Large angle of rotation (60 degrees)
      double theta = 1.04719755;

      // New coordinates
      double y = y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta);
      double z = z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta);

      // Rotate at right end
      values[0] = 0.0;
      values[1] = scale*(y - x[1]);
      values[2] = scale*(z - x[2]);
    }
  };

Next:

.. code-block:: c++

  int main()
  {

Inside the ``main`` function, we begin by defining a tetrahedral mesh
of the domain and the function space on this mesh. Here, we choose to
create a unit cube mesh with 17 ( = 16 + 1) vertices in each
direction. With this mesh, we initialize the (finite element) function
space defined by the generated code.

.. code-block:: c++

  // Create mesh and define function space
  UnitCube mesh (16, 16, 16);
  HyperElasticity::FunctionSpace V(mesh);

Now, the Dirichlet boundary conditions can be created using the class
``DirichletBC``, the previously initialized ``FunctionSpace`` ``V`` and
instances of the previously listed classes ``Left`` (for the left
boundary) and ``Right`` (for the right boundary), and ``Clamp`` (for
the value on the left boundary) and ``Rotation`` (for the value on the
right boundary).

.. code-block:: c++

  // Define Dirichlet boundaries
  Left left;
  Right right;

  // Define Dirichlet boundary functions
  Clamp c;
  Rotation r;

  // Create Dirichlet boundary conditions
  DirichletBC bcl(V, c, left);
  DirichletBC bcr(V, r, right);
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bcl); bcs.push_back(&bcr);

The two boundary conditions are collected in the container ``bcs``.

We use two instances of the class ``Constant`` to define the source
``B`` and the traction ``T``.

.. code-block:: c++

  // Define source and boundary traction functions
  Constant B(0.0, -0.5, 0.0);
  Constant T(0.1,  0.0, 0.0);

The solution for the displacement will be an instance of the class
``Function``, living in the function space ``V``; we define it here:

.. code-block:: c++

  // Define solution function
  Function u(V);

Next, we set the material parameters

.. code-block:: c++

  // Set material parameters
  const double E  = 10.0;
  const double nu = 0.3;
  Constant mu(E/(2*(1 + nu)));
  Constant lambda(E*nu/((1 + nu)*(1 - 2*nu)));

Now, we can initialize the bilinear and linear forms (:math:`a`,
:math:`L`) using the previously defined ``FunctionSpace`` ``V``. We
attach the material parameters and previously initialized functions to
the forms.

.. code-block:: c++

  // Create (linear) form defining (nonlinear) variational problem
  HyperElasticity::ResidualForm F(V);
  F.mu = mu; F.lmbda = lambda; F.B = B; F.T = T; F.u = u;

  // Create jacobian dF = F' (for use in nonlinear solver).
  HyperElasticity::JacobianForm J(V, V);
  J.mu = mu; J.lmbda = lambda; J.u = u;

Now, we have specified the variational forms and can consider the
solution of the variational problem.

.. code-block:: c++

  // Solve nonlinear variational problem F(u; v) = 0
  solve(F == 0, u, bcs, J);

Finally, the solution ``u`` is saved to a file named
``displacement.pvd`` in VTK format, and the displacement solution is
plotted.

.. code-block:: c++

  // Save solution in VTK format
  File file("displacement.pvd");
  file << u;

  // Plot solution
  plot(u);

  return 0;

Complete code
-------------

Complete UFL file
^^^^^^^^^^^^^^^^^

.. literalinclude:: HyperElasticity.ufl
   :start-after: # Compile this form with
   :language: python

Complete main file
^^^^^^^^^^^^^^^^^^

.. literalinclude:: main.cpp
   :start-after: // Begin demo
   :language: c++
