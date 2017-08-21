Stokes equations with Taylor-Hood quadrilateral elements (C++)
======================

It is assumed that the reader is familiar with :ref:`_demo_pde_stokes-taylor-hood-quadrilateral_python_documentation`.

This demo illustrates how to:

* Create quadrilateral mesh
* Define function spaces for quadrilateral mesh
* Use mixed function spaces

Implementation
--------------

The implementation is split in two files: a form file containing the
definition of the variational forms expressed in UFL and a C++ file
containing the actual solver.

Running this demo requires the files: :download:`main.cpp`,
:download:`Stokes.ufl`.


UFL form file
^^^^^^^^^^^^^

The UFL file is implemented in :download:`Stokes.ufl`, and the
explanation of the UFL file can be found at :doc:`here <Stokes.ufl>`.


C++ program
^^^^^^^^^^^

The main solver is implemented in the :download:`main.cpp` file.

At the top we include the DOLFIN header file and the generated header
file "Stokes.h" containing the variational forms for the Poisson
equation.  For convenience we also include the DOLFIN namespace. ::

.. code-block:: cpp

   #include <dolfin.h>
   #include "Stokes.h"

   using namespace dolfin;

The ``NoslipSubdomain`` and ``LidflowSubdomain`` are derived
from the :cpp:class:`SubDomain` class and define the part of the
boundary to which the Dirichlet boundary condition should be applied.

.. code-block:: cpp

   // Sub domain for no-slip Dirichlet boundary condition
   class NoslipSubdomain : public SubDomain
   {
     bool inside(const Array<double>& x, bool on_boundary) const
     {
       return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS;
     }
   };
   // Sub domain for no-slip Dirichlet boundary condition
   class LidflowSubdomain : public SubDomain
   {
     bool inside(const Array<double>& x, bool on_boundary) const
     {
       return x[1] > 1.0 - DOLFIN_EPS;
     }
   };

Inside the ``main`` function, we begin by defining a mesh of the
domain. As the unit square is a very standard domain, we can use a
built-in mesh provided by the :cpp:class:`UnitQuadMesh` factory. In
order to create a mesh consisting of 32 x 32 squares,
and the finite element space (specified in the form file)
defined relative to this mesh, we do as follows

.. code-block:: cpp

   int main()
   {
     // Create mesh and function space
     auto mesh = std::make_shared<Mesh>(UnitQuadMesh::create(32, 32));
     auto W = std::make_shared<Stokes::FunctionSpace>(mesh);

Now that we have our mixed function space we
define boundary conditions

.. code-block:: cpp

     // No-slip boundary condition for velocity
     // x0 = 0, x0 = 1, x1 = 0
     auto noslip = std::make_shared<Constant>(0.0, 0.0);
     auto noslip_boundary = std::make_shared<NoslipSubdomain>();
     DirichletBC bc0(W->sub(0), noslip, noslip_boundary);
     
     // Lid driven flow boundary condition for velocity
     // x1 = 1
     auto lidflow = std::make_shared<Constant>(1.0, 0.0);
     auto lidflow_boundary = std::make_shared<LidflowSubdomain>();
     DirichletBC bc1(W->sub(0), lidflow, lidflow_boundary);

     // Collect boundary conditions
     std::vector<const DirichletBC*> bcs = {{&bc0, &bc1}};

The bilinear and linear forms corresponding to the weak mixed
formulation of the Stokes equations are defined as follows

.. code-block:: cpp

     // Define variational problem
     auto f = std::make_shared<Constant>(0.0, 0.0);
     Stokes::BilinearForm a(W, W);
     Stokes::LinearForm L(W);
     L.f = f;

To compute the solution we use the bilinear and linear forms, and the
boundary condition, but we also need to create a :py:class:`Function
<dolfin.cpp.function.Function>` to store the solution(s). The (full)
solution will be stored in w, which we initialize using the mixed
function space ``W``. The actual
computation is performed by calling solve with the arguments ``a``,
``L``, ``w`` and ``bcs``.

.. code-block:: cpp

     // Compute solution
     Function w(W);
     solve(a == L, w, bcs);
     Function u = w[0];
     Function p = w[1];

Finally, we can store the solutions to files.

.. code-block:: cpp

     // Save solution in VTK format
     File ufile_pvd("velocity.pvd");
     ufile_pvd << u;
     File pfile_pvd("pressure.pvd");
     pfile_pvd << p;

     return 0;
   }
