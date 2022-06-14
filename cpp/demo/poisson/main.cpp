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
// :download:`poisson.py` and :download:`CMakeLists.txt`.
//
//
// UFL form file
// ^^^^^^^^^^^^^
//
// The UFL file is implemented in :download:`poisson.py`, and the
// explanation of the UFL file can be found at :doc:`here <poisson.py>`.
//
//
// C++ program
// ^^^^^^^^^^^
//
// The main solver is implemented in the :download:`main.cpp` file.
//
// At the top we include the DOLFINx header file and the generated header
// file "Poisson.h" containing the variational forms for the Poisson
// equation.  For convenience we also include the DOLFINx namespace.
//
// .. code-block:: cpp

#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using T = PetscScalar;

// Then follows the definition of the coefficient functions (for
// :math:`f` and :math:`g`), which are derived from the
// :cpp:class:`Expression` class in DOLFINx
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
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {32, 16},
        mesh::CellType::triangle, mesh::GhostMode::shared_facet));

    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Next, we define the variational formulation by initializing the
    // bilinear and linear forms (:math:`a`, :math:`L`) using the previously
    // defined :cpp:class:`FunctionSpace` ``V``.  Then we can create the
    // source and boundary flux term (:math:`f`, :math:`g`) and attach these
    // to the linear form.
    //
    // .. code-block:: cpp

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

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

    // Define boundary condition

    auto facets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto&& x) -> xt::xtensor<bool, 1>
        {
          auto x0 = xt::row(x, 0);
          return xt::isclose(x0, 0.0) or xt::isclose(x0, 2.0);
        });
    const auto bdofs = fem::locate_dofs_topological({*V}, 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    f->interpolate(
        [](auto&& x) -> xt::xarray<T>
        {
          auto dx = xt::square(xt::row(x, 0) - 0.5)
                    + xt::square(xt::row(x, 1) - 0.5);
          return 10 * xt::exp(-(dx) / 0.02);
        });

    g->interpolate(
        [](auto&& x) -> xt::xarray<T> { return xt::sin(5 * xt::row(x, 0)); });

    // Now, we have specified the variational forms and can consider the
    // solution of the variational problem. First, we need to define a
    // :cpp:class:`Function` ``u`` to store the solution. (Upon
    // initialization, it is simply set to the zero function.) Next, we can
    // call the ``solve`` function with the arguments ``a == L``, ``u`` and
    // ``bc`` as follows:
    //
    // .. code-block:: cpp

    // Compute solution
    fem::Function<T> u(V);
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                         {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting(b.mutable_array(), {a}, {{bc}}, {}, 1.0);
    b.scatter_rev(std::plus<T>());
    fem::set_bc(b.mutable_array(), {bc});

    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    lu.set_from_options();

    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u.x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    lu.solve(_u.vec(), _b.vec());

    // The function ``u`` will be modified during the call to solve. A
    // :cpp:class:`Function` can be saved to a file. Here, we output the
    // solution to a ``VTK`` file (specified using the suffix ``.pvd``) for
    // visualisation in an external program such as Paraview.
    //
    // .. code-block:: cpp

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({u}, 0.0);
  }

  PetscFinalize();

  return 0;
}
