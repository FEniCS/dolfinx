// # Biharmonic equation
//
// This demo illustrates how to:
//
// * Solve a fourth-order partial differential equation
// * Use an interior penalty (discontinuous Galerkin) method with
//   continuous Lagrange elements
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
// operator and $f$ is a prescribed source term. Here we use the boundary
// conditions
//
// \begin{align*}
//   u &= 0 \quad {\rm on} \ \partial\Omega, \\
//   \nabla^{2} u &= 0 \quad {\rm on} \ \partial\Omega.
// \end{align*}
//
// ### Weak formulation
//
// Multiplying the biharmonic equation by a test function and integrating
// by parts twice gives a problem with second-order derivatives, which
// requires $H^{2}$-conforming (roughly $C^{1}$-continuous) basis
// functions. To use Lagrange basis functions instead, the biharmonic
// equation can be split into two second-order equations (see the
// {doc}`mixed Poisson demo <demo_mixed_poisson>` for a mixed method for
// the Poisson equation), or continuity of the normal derivative between
// cells can be imposed weakly. This demo takes the second approach, using
// an interior penalty method.
//
// Let $\mathcal{T}$ be a triangulation of $\Omega$ with interior facets
// $\mathcal{E}_h^{\rm int}$. Functions evaluated on opposite sides of a
// facet are indicated by the subscripts $+$ and $-$. Using the
// continuous Lagrange space
//
// $$
// V = \left\{v \in H^{1}_{0}(\Omega)\,:\, v \in P_{k}(K) \
// \forall \ K \in \mathcal{T} \right\},
// $$
//
// the weak formulation reads: find $u \in V$ such that
//
// $$
// a(u,v)=L(v) \quad \forall \ v \in V,
// $$
//
// where
//
// \begin{align*}
// a(u, v) &=
// \sum_{K \in \mathcal{T}} \int_{K} \nabla^{2} u \nabla^{2} v \, {\rm d}x \\
// &\qquad+\sum_{E \in \mathcal{E}_h^{\rm int}}\left(\int_{E} \frac{\alpha}{h_E}
// [\!\![ \nabla u ]\!\!] [\!\![ \nabla v ]\!\!] \, {\rm d}s
// - \int_{E} \left<\nabla^{2} u \right>[\!\![ \nabla v ]\!\!]  \, {\rm d}s
// - \int_{E} [\!\![ \nabla u ]\!\!] \left<\nabla^{2} v \right> \,
// {\rm d}s\right), \\
// L(v) &= \int_{\Omega} fv \, {\rm d}x.
// \end{align*}
//
// Here $\left< w \right> = \frac{1}{2} (w_{+} + w_{-})$ is the average,
// $[\!\![ w ]\!\!] = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}$ is the jump
// in the normal component, $\alpha \ge 0$ is a penalty parameter and
// $h_E$ is the average diameter of the two cells sharing $E$. The
// condition $u = 0$ is imposed strongly through $V$, and
// $\nabla^{2} u = 0$ is imposed naturally by the formulation.
//
// This demo uses:
//
// * $\Omega = [0,1] \times [0,1]$
// * $k = 2$
// * $\alpha = 8$
// * $f = 4 \pi^4\sin(\pi x)\sin(\pi y)$, for which the exact solution is
//   $u = \sin(\pi x)\sin(\pi y)$
//
// ## Implementation
//
// The variational forms are expressed in UFL in a Python file, which
// FFCx compiles into C code. The C++ program builds the mesh and function
// space, assembles and solves the linear system, and writes the solution
// to file.
//
// Running this demo requires the files: {download}`demo_biharmonic/main.cpp`,
// {download}`demo_biharmonic/biharmonic.py` and
// {download}`demo_biharmonic/CMakeLists.txt`.
//
// ### UFL form file
//
// The UFL forms are defined in {download}`demo_biharmonic/biharmonic.py`.
// ````{admonition} UFL form implemented in Python
// :class: dropdown
// ![ufl-code]
// ````
//
// The supplied CMake configuration invokes FFCx with the scalar type of the
// installed PETSc. From a directory containing the downloaded files, configure
// and build the demo with
// ```bash
// cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer -B build -S .
// cmake --build build
// ```
//
// ### C++ program
//
// We include the DOLFINx headers and the header `biharmonic.h` generated
// by FFCx, which declares the compiled forms.

#include "biharmonic.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <memory>
#include <numbers>
#include <petscksp.h>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

// The `main` function performs the following steps:
//
// 1. Create a mesh of $\Omega$ with
//    {cpp:func}`dolfinx::mesh::create_rectangle`, using $32 \times 32$
//    squares each divided into two triangles, and a quadratic Lagrange
//    {cpp:class}`dolfinx::fem::FunctionSpace` on it.
// 2. Interpolate the source term $f$, and create the forms $a$ and $L$
//    with the penalty parameter $\alpha$ as a
//    {cpp:class}`dolfinx::fem::Constant`.
// 3. Create a {cpp:class}`dolfinx::fem::DirichletBC` that sets the
//    degrees-of-freedom on the boundary facets to zero.
// 4. Assemble the linear system, solve it with a direct (LU) solver, and
//    write the solution to file.

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  dolfinx::common::petsc::check(PetscInitialize(&argc, &argv, nullptr, nullptr),
                                "PetscInitialize");
  {
    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 32},
        mesh::CellType::triangle, graph::partition_graph,
        mesh::DiagonalType::right, 2, mesh::GhostMode::shared_facet));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::triangle, 2,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    // Create function space
    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      element, mesh->geometry().dim())));

    // Create the source term f and the penalty parameter alpha
    auto f = std::make_shared<fem::Function<T>>(V);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f(x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto pi = std::numbers::pi;
            f[p] = 4.0 * std::pow(pi, 4) * std::sin(pi * x(0, p))
                   * std::sin(pi * x(1, p));
          }
          return {std::move(f), std::vector<std::size_t>{f.size()}};
        });
    auto alpha = std::make_shared<fem::Constant<T>>(8.0);

    // Define variational forms
    fem::Form<T> a = fem::create_form<T>(*form_biharmonic_a, {V, V}, {},
                                         {{"alpha", alpha}}, {}, {});
    fem::Form<T> L
        = fem::create_form<T>(*form_biharmonic_L, {V}, {{"f", f}}, {}, {}, {});

    // Define the boundary condition u = 0 on all boundary facets
    auto facets = mesh::exterior_facet_indices(*mesh->topology());
    const auto bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    fem::DirichletBC<T> bc(0, bdofs, V);

    // Assemble the matrix. Rows and columns of constrained
    // degrees-of-freedom are zeroed and one is placed on the diagonal.
    fem::Function<T> u(V);
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(a), false);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    common::petsc::check(MatZeroEntries(A.mat()), "MatZeroEntries");
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         a, {bc});
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), ADD_VALUES), *V,
                         {bc});
    common::petsc::check(MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY),
                         "MatAssemblyBegin");
    common::petsc::check(MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY),
                         "MatAssemblyEnd");

    // Assemble the right-hand side vector, modify it to account for the
    // Dirichlet condition, and set the constrained entries to the
    // boundary values
    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);

    // Solve the linear system with a direct LU solver
    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    common::petsc::set_option("ksp_type", "preonly");
    common::petsc::set_option("pc_type", "lu");
    lu.set_from_options();

    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u.x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    if (lu.solve(_u.vec(), _b.vec()) < 0)
      throw std::runtime_error("Linear solver did not converge.");

    // The KSP object is available for anything the solver does not
    // wrap, here the number of linear solver iterations
    PetscInt num_it = 0;
    common::petsc::check(KSPGetIterationNumber(lu.ksp(), &num_it),
                         "KSPGetIterationNumber");
    std::cout << "Number of linear solver iterations: " << num_it << std::endl;

    // Update ghost values before output
    u.x()->scatter_fwd();

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({u}, 0);
  }

  common::petsc::check(PetscFinalize(), "PetscFinalize");
  return 0;
}
