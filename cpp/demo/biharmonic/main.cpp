// Biharmonic equation (C++)
// =========================
//
// This demo illustrates how to:
//
// * Solve a linear partial differential equation
// * Use a discontinuous Galerkin method
// * Solve a fourth-order differential equation
//
// The solution for :math:`u` in this demo will look as follows:
//
// .. image:: ../biharmonic_u.png
//     :scale: 75 %
//
// Equation and problem definition
// -------------------------------
//
// The biharmonic equation is a fourth-order elliptic equation. On the
// domain :math:`\Omega \subset \mathbb{R}^{d}`, :math:`1 \le d \le 3`,
// it reads
//
// .. math::
//    \nabla^{4} u = f \quad {\rm in} \ \Omega,
//
// where :math:`\nabla^{4} \equiv \nabla^{2} \nabla^{2}` is the
// biharmonic operator and :math:`f` is a prescribed source term. To
// formulate a complete boundary value problem, the biharmonic equation
// must be complemented by suitable boundary conditions.
//
// Multiplying the biharmonic equation by a test function and integrating
// by parts twice leads to a problem second-order derivatives, which
// would requires :math:`H^{2}` conforming (roughly :math:`C^{1}`
// continuous) basis functions.  To solve the biharmonic equation using
// Lagrange finite element basis functions, the biharmonic equation can
// be split into two second-order equations (see the Mixed Poisson demo
// for a mixed method for the Poisson equation), or a variational
// formulation can be constructed that imposes weak continuity of normal
// derivatives between finite element cells.  The demo uses a
// discontinuous Galerkin approach to impose continuity of the normal
// derivative weakly.
//
// Consider a triangulation :math:`\mathcal{T}` of the domain
// :math:`\Omega`, where the set of interior facets is denoted by
// :math:`\mathcal{E}_h^{\rm int}`.  Functions evaluated on opposite
// sides of a facet are indicated by the subscripts ':math:`+`' and
// ':math:`-`'.  Using the standard continuous Lagrange finite element
// space
//
// .. math::
//     V = \left\{v \in H^{1}_{0}(\Omega)\,:\, v \in P_{k}(K) \ \forall \ K \in \mathcal{T} \right\}
//
// and considering the boundary conditions
//
// .. math::
//    u            &= 0 \quad {\rm on} \ \partial\Omega \\
//    \nabla^{2} u &= 0 \quad {\rm on} \ \partial\Omega
//
// a weak formulation of the biharmonic problem reads: find :math:`u \in
// V` such that
//
// .. math::
//   a(u,v)=L(v) \quad \forall \ v \in V,
//
// where the bilinear form is
//
// .. math::
//    a(u, v) = \sum_{K \in \mathcal{T}} \int_{K} \nabla^{2} u \nabla^{2} v \, {\rm d}x \
//   +\sum_{E \in \mathcal{E}_h^{\rm int}}\left(\int_{E} \frac{\alpha}{h_E} [\!\![ \nabla u ]\!\!] [\!\![ \nabla v ]\!\!] \, {\rm d}s
//   - \int_{E} \left<\nabla^{2} u \right>[\!\![ \nabla v ]\!\!]  \, {\rm d}s
//   - \int_{E} [\!\![ \nabla u ]\!\!]  \left<\nabla^{2} v \right>  \, {\rm d}s\right)
//
// and the linear form is
//
// .. math::
//   L(v) = \int_{\Omega} fv \, {\rm d}x
//
// Furthermore, :math:`\left< u \right> = \frac{1}{2} (u_{+} + u_{-})`, :math:`[\!\![
// w ]\!\!]  = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}`, :math:`\alpha \ge
// 0` is a penalty parameter and :math:`h_E` is a measure of the cell size.
//
// The input parameters for this demo are defined as follows:
//
// * :math:`\Omega = [0,1] \times [0,1]` (a unit square)
// * :math:`\alpha = 8.0` (penalty parameter)
// * :math:`f = 4.0 \pi^4\sin(\pi x)\sin(\pi y)` (source term)
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
// :download:`biharmonic.py` and :download:`CMakeLists.txt`.
//
// UFL form file
// ^^^^^^^^^^^^^
//
// The UFL file is implemented in :download:`biharmonic.py`, and the
// explanation of the UFL file can be found at :doc:`here
// <biharmonic.py>`.
//
//
// C++ program
// ^^^^^^^^^^^
//
// The main solver is implemented in the :download:`main.cpp` file.
//
// At the top we include the DOLFINx header file and the generated header
// file "biharmonic.h" containing the variational forms for the
// Biharmonic equation, which are defined in the UFL form file.
// For convenience we also include the DOLFINx namespace.
//
// .. code-block:: cpp

#include "biharmonic.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;

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
    // Create mesh
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh>(
        mesh::create_rectangle(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                               {32, 32}, mesh::CellType::triangle, part));

    // A function space object, which is defined in the generated code, is
    // created:
    //
    // .. code-block:: cpp

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace>(
    fem::create_functionspace(functionspace_form_biharmonic_a, "u", mesh));

    // The source function ::math:`f` and the penalty term ::math:`\alpha`
    // are declared:
    //
    // .. code-block:: cpp

    auto f = std::make_shared<fem::Function<T>>(V);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto pi = std::numbers::pi;
            f.push_back(4.0*std::pow(pi, 4)*
              std::sin(pi*x(0,p))*std::sin(pi*x(1,p)));
          }

          return {f, {f.size()}};
        });
    auto alpha = std::make_shared<fem::Constant<T>>(8.0);
    // Define variational forms
    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_biharmonic_a, {V, V}, {}, {{"alpha", alpha}}, {}));
    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_biharmonic_L, {V}, {{"f", f}}, {}, {}));

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
        [](auto x)
        {
          constexpr double eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            double x0 = x(0, p);
            double x1 = x(1, p);
            if (std::abs(x0) < eps or std::abs(x0 - 1) < eps)
              marker[p] = true;
            if (std::abs(x1) < eps or std::abs(x1 - 1) < eps)
              marker[p] = true;
          }
          return marker;
        });
    const auto bdofs = fem::locate_dofs_topological({*V}, 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

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
