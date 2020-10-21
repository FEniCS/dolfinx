// Poisson equation (C++)
// ======================
//
// This demo illustrates how to:
//
// * Solve a linear partial differential equation
// * Create and apply Dirichlet boundary conditions
// * Define Expressions
// * Define a FunctionSpace
//
// The solution for :math:`u` in this demo will look as follows:
//
// .. image:: ../poisson_u.png
//     :scale: 75 %
//
//
// Equation and problem definition
// -------------------------------
//
// The Poisson equation is the canonical elliptic partial differential
// equation.  For a domain :math:`\Omega \subset \mathbb{R}^n` with
// boundary :math:`\partial \Omega = \Gamma_{D} \cup \Gamma_{N}`, the
// Poisson equation with particular boundary conditions reads:
//
// .. math::
//    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
//      u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
//      \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
//
// Here, :math:`f` and :math:`g` are input data and :math:`n` denotes the
// outward directed boundary normal. The most standard variational form
// of Poisson equation reads: find :math:`u \in V` such that
//
// .. math::
//    a(u, v) = L(v) \quad \forall \ v \in V,
//
// where :math:`V` is a suitable function space and
//
// .. math::
//    a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
//    L(v)    &= \int_{\Omega} f v \, {\rm d} x
//    + \int_{\Gamma_{N}} g v \, {\rm d} s.
//
// The expression :math:`a(u, v)` is the bilinear form and :math:`L(v)`
// is the linear form. It is assumed that all functions in :math:`V`
// satisfy the Dirichlet boundary conditions (:math:`u = 0 \ {\rm on} \
// \Gamma_{D}`).
//
// In this demo, we shall consider the following definitions of the input
// functions, the domain, and the boundaries:
//
// * :math:`\Omega = [0,1] \times [0,1]` (a unit square)
// * :math:`\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}`
// (Dirichlet boundary)
// * :math:`\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}`
// (Neumann boundary)
// * :math:`g = \sin(5x)` (normal derivative)
// * :math:`f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)` (source term)
//
//
// Implementation
// --------------
//
// The implementation is split in two files: a form file containing the
// definition of the variational forms expressed in UFL and a C++ file
// containing the actual solver.
//
// Running this demo requires the files: :download:`main.cpp`,
// :download:`Poisson.ufl` and :download:`CMakeLists.txt`.
//
//
// UFL form file
// ^^^^^^^^^^^^^
//
// The UFL file is implemented in :download:`Poisson.ufl`, and the
// explanation of the UFL file can be found at :doc:`here <Poisson.ufl>`.
//
//
// C++ program
// ^^^^^^^^^^^
//
// The main solver is implemented in the :download:`main.cpp` file.
//
// At the top we include the DOLFIN header file and the generated header
// file "Poisson.h" containing the variational forms for the Poisson
// equation.  For convenience we also include the DOLFIN namespace.
//
// .. code-block:: cpp

#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/function/Constant.h>

using namespace dolfinx;

// Then follows the definition of the coefficient functions (for
// :math:`f` and :math:`g`), which are derived from the
// :cpp:class:`Expression` class in DOLFIN
//
// .. code-block:: cpp

// Inside the ``main`` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the :cpp:class:`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified in
// the form file) defined relative to this mesh, we do as follows
//
// .. code-block:: cpp

int main(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  {
    // Create mesh and function space
    auto cmap = fem::create_coordinate_map(create_coordinate_map_poisson);
    auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
        MPI_COMM_WORLD,
        {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 0.0)},
        {32, 32}, cmap, mesh::GhostMode::none));

    auto V = fem::create_functionspace(create_functionspace_form_poisson_a, "u",
                                       mesh);

    // Next, we define the variational formulation by initializing the
    // bilinear and linear forms (:math:`a`, :math:`L`) using the previously
    // defined :cpp:class:`FunctionSpace` ``V``.  Then we can create the
    // source and boundary flux term (:math:`f`, :math:`g`) and attach these
    // to the linear form.
    //
    // .. code-block:: cpp

    // Define variational forms
    auto a = fem::create_form<PetscScalar>(create_form_poisson_a, {V, V});
    auto L = fem::create_form<PetscScalar>(create_form_poisson_L, {V});

    auto f = std::make_shared<function::Function<PetscScalar>>(V);
    auto g = std::make_shared<function::Function<PetscScalar>>(V);

    // Now, the Dirichlet boundary condition (:math:`u = 0`) can be created
    // using the class :cpp:class:`DirichletBC`. A :cpp:class:`DirichletBC`
    // takes two arguments: the value of the boundary condition,
    // and the part of the boundary on which the condition applies.
    // In our example, the value of the boundary condition (0.0) can
    // represented using a :cpp:class:`Function`, and the Dirichlet boundary
    // is defined by the indices of degrees of freedom to which the boundary
    // condition applies.
    // The definition of the Dirichlet boundary condition then looks
    // as follows:
    //
    // .. code-block:: cpp

    // FIXME: zero function and make sure ghosts are updated
    // Define boundary condition
    auto u0 = std::make_shared<function::Function<PetscScalar>>(V);

    const auto bdofs = fem::locate_dofs_geometrical({*V}, [](auto& x) {
      static const double epsilon = std::numeric_limits<double>::epsilon();
      return (x.row(0).abs() < 10.0 * epsilon
              or (x.row(0) - 1.0).abs() < 10.0 * epsilon);
    });

    std::vector bc{
        std::make_shared<const fem::DirichletBC<PetscScalar>>(u0, bdofs)};

    f->interpolate([](auto& x) {
      auto dx = Eigen::square(x - 0.5);
      return 10.0 * Eigen::exp(-(dx.row(0) + dx.row(1)) / 0.02);
    });

    g->interpolate([](auto& x) { return Eigen::sin(5 * x.row(0)); });
    L->set_coefficients({{"f", f}, {"g", g}});

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<function::Constant<PetscScalar>>(2.0);
    a->set_constants({{"kappa", kappa}});

    // Now, we have specified the variational forms and can consider the
    // solution of the variational problem. First, we need to define a
    // :cpp:class:`Function` ``u`` to store the solution. (Upon
    // initialization, it is simply set to the zero function.) Next, we can
    // call the ``solve`` function with the arguments ``a == L``, ``u`` and
    // ``bc`` as follows:
    //
    // .. code-block:: cpp

    // Compute solution
    function::Function<PetscScalar> u(V);
    la::PETScMatrix A = fem::create_matrix(*a);
    la::PETScVector b(*L->function_space(0)->dofmap()->index_map);

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::PETScMatrix::add_fn(A.mat()), *a, bc);
    fem::add_diagonal(la::PETScMatrix::add_fn(A.mat()), *V, bc);
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    VecSet(b.vec(), 0.0);
    VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
    fem::assemble_vector_petsc(b.vec(), *L);
    fem::apply_lifting_petsc(b.vec(), {a}, {{bc}}, {}, 1.0);
    VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
    VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
    fem::set_bc_petsc(b.vec(), bc, nullptr);

    la::PETScKrylovSolver lu(MPI_COMM_WORLD);
    la::PETScOptions::set("ksp_type", "preonly");
    la::PETScOptions::set("pc_type", "lu");
    lu.set_from_options();

    lu.set_operator(A.mat());
    lu.solve(u.vector(), b.vec());

    // The function ``u`` will be modified during the call to solve. A
    // :cpp:class:`Function` can be saved to a file. Here, we output the
    // solution to a ``VTK`` file (specified using the suffix ``.pvd``) for
    // visualisation in an external program such as Paraview.
    //
    // .. code-block:: cpp

    // Save solution in VTK format
    io::VTKFile file("u.pvd");
    file.write(u);
  }

  common::SubSystemsManager::finalize_petsc();
  return 0;
}
