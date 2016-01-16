.. Documentation for the incompressible Navier-Stokes demo from DOLFIN.

.. _demo_pde_navier_stokes_cpp_documentation:

Incompressible Navier-Stokes equations
======================================

.. include:: ../common.txt

Implementation
--------------

The implementation is split in four files: three form files containing the
definition of the variational forms expressed in UFL and a C++ file
containing the actual solver.

Running this demo requires the files: :download:`main.cpp`,
:download:`TentativeVelocity.ufl`, :download:`VelocityUpdate.ufl`,
:download:`PressureUpdate.ufl` and :download:`CMakeLists.txt`.

UFL form files
^^^^^^^^^^^^^^

The variational forms for the three steps of Chorin's method are
implemented in three separate UFL form files.

The variational problem for the tentative velocity is implemented as
follows:

.. code-block:: python

    # Define function spaces (P2-P1)
    V = VectorElement("Lagrange", triangle, 2)
    Q = FiniteElement("Lagrange", triangle, 1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define coefficients
    k  = Constant(triangle)
    u0 = Coefficient(V)
    f  = Coefficient(V)
    nu = 0.01

    # Define bilinear and linear forms
    eq = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
        nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
    a = lhs(eq)
    L = rhs(eq)

The variational problem for the pressure update is implemented as
follows:

.. code-block:: python

    # Define function spaces (P2-P1)
    V = VectorElement("Lagrange", triangle, 2)
    Q = FiniteElement("Lagrange", triangle, 1)

    # Define trial and test functions
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define coefficients
    k  = Constant(triangle)
    u1 = Coefficient(V)

    # Define bilinear and linear forms
    a = inner(grad(p), grad(q))*dx
    L = -(1/k)*div(u1)*q*dx

The variational problem for the velocity update is implemented as
follows:

.. code-block:: python

    # Define function spaces (P2-P1)
    V = VectorElement("Lagrange", triangle, 2)
    Q = FiniteElement("Lagrange", triangle, 1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define coefficients
    k  = Constant(triangle)
    u1 = Coefficient(V)
    p1 = Coefficient(Q)

    # Define bilinear and linear forms
    a = inner(u, v)*dx
    L = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

Before the form files can be used in the C++ program, they must be
compiled using FFC:

.. code-block:: sh

    ffc -l dolfin TentativeVelocity.ufl
    ffc -l dolfin VelocityUpdate.ufl
    ffc -l dolfin PressureUpdate.ufl

Note the flag ``-l dolfin`` which tells FFC to generate
DOLFIN-specific wrappers that make it easy to access the generated
code from within DOLFIN.

C++ program
^^^^^^^^^^^

In the C++ program, :download:`main.cpp`, we start by including
``dolfin.h`` and the generated header files:

.. code-block:: c++

    #include <dolfin.h>
    #include "TentativeVelocity.h"
    #include "PressureUpdate.h"
    #include "VelocityUpdate.h"

To be able to use classes and functions from the DOLFIN namespace
directly, we write

.. code-block:: c++

    using namespace dolfin;

Next, we define the subdomains that we will use to specify boundary
conditions. We do this by defining subclasses of
:cpp:class:`SubDomain` and overloading the function ``inside``:

.. code-block:: c++

    // Define noslip domain
    class NoslipDomain : public SubDomain
    {
      bool inside(const Array<double>& x, bool on_boundary) const
      {
        return (on_boundary &&
                (x[0] < DOLFIN_EPS || x[1] < DOLFIN_EPS ||
                 (x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS)));
      }
    };

    // Define inflow domain
    class InflowDomain : public SubDomain
    {
      bool inside(const Array<double>& x, bool on_boundary) const
      { return x[1] > 1.0 - DOLFIN_EPS; }
    };

    // Define inflow domain
    class OutflowDomain : public SubDomain
    {
      bool inside(const Array<double>& x, bool on_boundary) const
      { return x[0] > 1.0 - DOLFIN_EPS; }
    };

We also define a subclass of :cpp:class:`Expression` which we will use
to specify the time-dependent boundary value for the pressure at the
inflow.

.. code-block:: c++

    // Define pressure boundary value at inflow
    class InflowPressure : public Expression
    {
    public:

      // Constructor
      InflowPressure() : t(0) {}

      // Evaluate pressure at inflow
      void eval(Array<double>& values, const Array<double>& x) const
      { values[0] = sin(3.0*t); }

      // Current time
      double t;

    };

Note that the member variable ``t`` is not automatically updated
during time-stepping, so we must remember to manually update the value
of the current time in each time step.

Once we have defined all classes we will use to write our program, we
start our C++ program by writing

.. code-block:: c++

    int main()
    {

For the parallel case, we turn off log messages from processes other than
the the root process to avoid excessive output:

.. code-block:: c++

    // Print log messages only from the root process in parallel
    parameters["std_out_all_processes"] = false;

We then load the mesh for the L-shaped domain from file:

.. code-block:: c++

    // Load mesh from file
    auto mesh = std::make_shared<Mesh>("../lshape.xml.gz");

We next define a pair of function spaces :math:`V` and :math:`Q` for
the velocity and pressure, and test and trial functions on these
spaces:

.. code-block:: c++

    // Create function spaces
    auto V = std::make_shared<VelocityUpdate::FunctionSpace>(mesh);
    auto Q = std::make_shared<PressureUpdate::FunctionSpace>(mesh);

The time step and the length of the interval are defined by:

.. code-block:: c++

    // Set parameter values
    double dt = 0.01;
    double T = 3;

We next define the time-dependent pressure boundary value, and zero
scalar and vector constants that will be used for boundary conditions
below.

.. code-block:: c++

   // Define values for boundary conditions
   auto p_in = std::make_shared<InflowPressure>();
   auto zero = std::make_shared<Constant>(0.0);
   auto zero_vector = std::make_shared<Constant>(0.0, 0.0);

Before we can define our boundary conditions, we also need to
instantiate the classes we defined above for the boundary subdomains:

.. code-block:: c++

    // Define subdomains for boundary conditions
    auto noslip_domain = std::make_shared<NoslipDomain>();
    auto inflow_domain = std::make_shared<InflowDomain>();
    auto outflow_domain = std::make_shared<OutflowDomain>() ;

We may now define the boundary conditions for the velocity and
pressure. We define one no-slip boundary condition for the velocity
and a pair of boundary conditions for the pressure at the inflow and
outflow boundaries:

.. code-block:: c++

    // Define boundary conditions
    DirichletBC noslip(V, zero_vector, noslip_domain);
    DirichletBC inflow(Q, p_in, inflow_domain);
    DirichletBC outflow(Q, zero, outflow_domain);
    std::vector<DirichletBC*> bcu = {&noslip};
    std::vector<DirichletBC*> bcp = {{&inflow, &outflow}};

We collect the boundary conditions in the two arrays ``bcu`` and
``bcp`` so that we may easily iterate over them below when we apply
the boundary conditions. This makes it easy to add new boundary
conditions or use this demo program to solve the Navier-Stokes
equations on other geometries.

We next define the functions and the coefficients that will be used
below:

.. code-block:: c++

    // Create functions
    auto u0 = std::make_shared<Function>(V);
    auto u1 = std::make_shared<Function>(V);
    auto p1 = std::make_shared<Function>(Q);

    // Create coefficients
    auto k = std::make_shared<Constant>(dt);
    auto f = std::make_shared<Constant>(0, 0);

The next step is now to define the variational problems for the three
steps of Chorin's method. We do this by instantiating the classes
generated from our UFL form files:

.. code-block:: c++

   // Create forms
   TentativeVelocity::BilinearForm a1(V, V);
   TentativeVelocity::LinearForm L1(V);
   PressureUpdate::BilinearForm a2(Q, Q);
   PressureUpdate::LinearForm L2(Q);
   VelocityUpdate::BilinearForm a3(V, V);
   VelocityUpdate::LinearForm L3(V);

Since the forms depend on coefficients, we have to attach the
coefficients defined above to the appropriate forms:

.. code-block:: c++

  // Set coefficients
  a1.k = k; L1.k = k; L1.u0 = u0; L1.f = f;
  L2.k = k; L2.u1 = u1;
  L3.k = k; L3.u1 = u1; L3.p1 = p1;

Since the bilinear forms do not depend on any coefficients that change
during time-stepping, the corresponding matrices remain constant. We
may therefore assemble these before the time-stepping begins:

.. code-block:: c++

    // Assemble matrices
    Matrix A1, A2, A3;
    assemble(A1, a1);
    assemble(A2, a2);
    assemble(A3, a3);

    // Create vectors
    Vector b1, b2, b3;

    // Use amg preconditioner if available
    const std::string prec(has_krylov_solver_preconditioner("amg") ? "amg" : "default");

We also created the vectors that will be used below to assemble
right-hand sides.

During time-stepping, we will store the solution in VTK format
(readable by MayaVi and Paraview). We therefore create a pair of files
that can be used to store the solution. Specifying the ``.pvd`` suffix
signals that the solution should be stored in VTK format:

.. code-block:: c++

    // Create files for storing solution
    File ufile("results/velocity.pvd");
    File pfile("results/pressure.pvd");

The time-stepping loop is now implemented as follows:

.. code-block:: c++

    // Time-stepping
    double t = dt;
    while (t < T + DOLFIN_EPS)
    {
      // Update pressure boundary condition
      p_in->t = t;

We remember to update the current time for the time-dependent pressure
boundary value.

For each of the three steps of Chorin's method, we assemble the
right-hand side, apply boundary conditions, and solve a linear
system. Note the different use of preconditioners. Incomplete LU
factorization is used for the computation of the tentative velocity
and the velocity update, while algebraic multigrid is used for the
pressure equation if available:

.. code-block:: c++

    // Compute tentative velocity step
    begin("Computing tentative velocity");
    assemble(b1, L1);
    for (std::size_t i = 0; i < bcu.size(); i++)
      bcu[i]->apply(A1, b1);
    solve(A1, *u1->vector(), b1, "gmres", "default");
    end();

    // Pressure correction
    begin("Computing pressure correction");
    assemble(b2, L2);
    for (std::size_t i = 0; i < bcp.size(); i++)
    {
      bcp[i]->apply(A2, b2);
      bcp[i]->apply(*p1->vector());
    }
    solve(A2, *p1->vector(), b2, "bicgstab", prec);
    end();

    // Velocity correction
    begin("Computing velocity correction");
    assemble(b3, L3);
    for (std::size_t i = 0; i < bcu.size(); i++)
      bcu[i]->apply(A3, b3);
    solve(A3, *u1->vector(), b3, "gmres", "default");
    end();

Note the use of ``begin`` and ``end``; these improve the readability
of the output from the program by adding indentation to diagnostic
messages.

At the end of the time-stepping loop, we store the solution to file
and update values for the next time step:

.. code-block:: c++

    // Save to file
    ufile << *u1;
    pfile << *p1;

    // Move to next time step
    *u0 = *u1;
    t += dt;

Finally, we plot the solution and the program is finished:

.. code-block:: c++

    // Plot solution
    plot(p1, "Pressure");
    plot(u1, "Velocity");
    interactive();

    return 0;
  }



Complete code
-------------

Complete UFL files
^^^^^^^^^^^^^^^^^^

.. literalinclude:: TentativeVelocity.ufl
   :start-after: # Compile
   :language: python


.. literalinclude:: VelocityUpdate.ufl
   :start-after: # Compile
   :language: python


.. literalinclude:: PressureUpdate.ufl
   :start-after: # Compile
   :language: python

Complete main file
^^^^^^^^^^^^^^^^^^

.. literalinclude:: main.cpp
   :start-after: // Begin demo
   :language: c++
