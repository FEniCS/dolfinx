// # Biharmonic equation
//
// This demo illustrates how to:
//
// * Solve a linear partial differential equation
// * Use a discontinuous Galerkin method
// * Solve a fourth-order differential equation
//
// ## Equation and problem definition
//
// ### Strong formulation
//
// The biharmonic equation is a fourth-order elliptic equation. On the
// domain $\Omega \subset \mathbb{R}^{d}$, $1 \le d \le 3$, it reads
//
// $$
// \nabla^{4} u = f \quad {\rm in} \ \Omega,
// $$
//
// where $\nabla^{4} \equiv \nabla^{2} \nabla^{2}$ is the biharmonic
// operator and $f$ is a prescribed source term. To formulate a complete
// boundary value problem, the biharmonic equation must be complemented
// by suitable boundary conditions.
//
// ### Weak formulation
//
// Multiplying the biharmonic equation by a test function and integrating
// by parts twice leads to a problem of second-order derivatives, which would
// require $H^{2}$ conforming (roughly $C^{1}$ continuous) basis functions.
// To solve the biharmonic equation using Lagrange finite element basis
// functions, the biharmonic equation can be split into two second-order
// equations (see the Mixed Poisson demo for a mixed method for the Poisson
// equation), or a variational formulation can be constructed that imposes
// weak continuity of normal derivatives between finite element cells.
// This demo uses a discontinuous Galerkin approach to impose continuity
// of the normal derivative weakly.
//
// Consider a triangulation $\mathcal{T}$ of the domain $\Omega$, where
// the set of interior facets is denoted by $\mathcal{E}_h^{\rm int}$.
// Functions evaluated on opposite sides of a facet are indicated by the
// subscripts $+$ and $-$.
// Using the standard continuous Lagrange finite element space
//
// $$
// V = \left\{v \in H^{1}_{0}(\Omega)\,:\, v \in P_{k}(K) \
// \forall \ K \in \mathcal{T} \right\}
// $$
//
// and considering the boundary conditions
//
// \begin{align}
// u &= 0 \quad {\rm on} \ \partial\Omega, \\
// \nabla^{2} u &= 0 \quad {\rm on} \ \partial\Omega,
// \end{align}
//
// a weak formulation of the biharmonic problem reads: find $u \in V$ such that
//
// $$
// a(u,v)=L(v) \quad \forall \ v \in V,
// $$
//
// where the bilinear form is
//
// \begin{align*}
// a(u, v) &=
// \sum_{K \in \mathcal{T}} \int_{K} \nabla^{2} u \nabla^{2} v \, {\rm d}x \\
// &\qquad+\sum_{E \in \mathcal{E}_h^{\rm int}}\left(\int_{E} \frac{\alpha}{h_E}
// [\!\![ \nabla u ]\!\!] [\!\![ \nabla v ]\!\!] \, {\rm d}s
// - \int_{E} \left<\nabla^{2} u \right>[\!\![ \nabla v ]\!\!]  \, {\rm d}s
// - \int_{E} [\!\![ \nabla u ]\!\!] \left<\nabla^{2} v \right> \,
// {\rm d}s\right)
// \end{align*}
//
// and the linear form is
//
// $$
// L(v) = \int_{\Omega} fv \, {\rm d}x.
// $$
//
// Furthermore, $\left< u \right> = \frac{1}{2} (u_{+} + u_{-})$,
// $[\!\![ w ]\!\!]  = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}$,
// $\alpha \ge 0$ is a penalty parameter and
// $h_E$ is a measure of the cell size.
//
// The input parameters for this demo are defined as follows:
//
// - $\Omega = [0,1] \times [0,1]$ (a unit square)
// - $\alpha = 8.0$ (penalty parameter)
// - $f = 4.0 \pi^4\sin(\pi x)\sin(\pi y)$ (source term)
//
//
//
// ## Implementation
//
// The implementation is in two files: a form file containing the
// definition of the variational forms expressed in UFL and a C++ file
// containing the actual solver.
//
// Running this demo requires the files: {download}`demo_biharmonic/main.cpp`,
// {download}`demo_biharmonic/biharmonic.py` and
// {download}`demo_biharmonic/CMakeLists.txt`.
//
// ### UFL form file
//
// The UFL file is implemented in {download}`demo_biharmonic/biharmonic.py`.
// ````{admonition} UFL form implemented in python
// :class: dropdown
// ![ufl-code]
// ````
//
// ````{note}
// TODO: explanation on how to run cmake and/or shell commands for `ffcx`.
// To compile biharmonic.py using FFCx with an option
// for PETSc scalar type `float64` one would execute the command
// ```bash
// ffcx biharmonic.py --scalar_type=float64
// ```
// ````
//
// ### C++ program
//
// The main solver is implemented in the {download}`demo_biharmonic/main.cpp`
// file.
//
// At the top we include the DOLFINx header file and the generated
// header file "biharmonic.h" containing the variational forms for the
// Biharmonic equation, which are defined in the UFL form file. For
// convenience we also include the DOLFINx namespace.

#include "biharmonic.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <numbers>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

// Inside the `main` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the {cpp:class}`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified
// in the form file) defined relative to this mesh, we do as follows

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    //  Create mesh
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                  {32, 32}, mesh::CellType::triangle, part));

    //    A function space object, which is defined in the generated code,
    //    is created:

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::triangle, 2,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    //  Create function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element));

    // The source function $f$ and the penalty term $\alpha$ are
    // declared:
    auto f = std::make_shared<fem::Function<T>>(V);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto pi = std::numbers::pi;
            f.push_back(4.0 * std::pow(pi, 4) * std::sin(pi * x(0, p))
                        * std::sin(pi * x(1, p)));
          }
          return {f, {f.size()}};
        });
    auto alpha = std::make_shared<fem::Constant<T>>(8.0);

    //  Define variational forms
    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_biharmonic_a, {V, V}, {}, {{"alpha", alpha}}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_biharmonic_L, {V}, {{"f", f}}, {}, {}));

    //  Now, the Dirichlet boundary condition ($u = 0$) can be
    //  created using the class {cpp:class}`DirichletBC`. A
    //  {cpp:class}`DirichletBC` takes two arguments: the value of the
    //  boundary condition, and the part of the boundary on which the
    //  condition applies. In our example, the value of the boundary
    //  condition (0.0) can represented using a {cpp:class}`Function`,
    //  and the Dirichlet boundary is defined by the indices of degrees
    //  of freedom to which the boundary condition applies. The
    //  definition of the Dirichlet boundary condition then looks as
    //  follows:

    //  Define boundary condition
    auto facets = mesh::locate_entities_boundary(*mesh, 1);
    const auto bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    //  Now, we have specified the variational forms and can consider
    //  the solution of the variational problem. First, we need to
    //  define a {cpp:class}`Function` `u` to store the solution. (Upon
    //  initialization, it is simply set to the zero function.) Next, we
    //  can call the `solve` function with the arguments `a == L`, `u`
    //  and `bc` as follows:

    //  Compute solution
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
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1.0));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, U>(b.mutable_array(), {bc});

    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    lu.set_from_options();

    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u.x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    lu.solve(_u.vec(), _b.vec());

    //  Update ghost values before output
    u.x()->scatter_fwd();

    // The function `u` will be modified during the call to solve. A
    // {cpp:class}`Function` can be saved to a file. Here, we output the
    // solution to a `VTK` file (specified using the suffix `.pvd`) for
    // visualisation in an external program such as Paraview.

    //  Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({u}, 0.0);
  }

  PetscFinalize();
  return 0;
}
