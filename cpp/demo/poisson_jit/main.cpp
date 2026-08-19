// # Poisson equation with just-in-time compiled forms
//
// This demo solves the same problem as {download}`demo_poisson`, but
// rather than compiling the variational forms ahead of time with FFCx
// and linking the generated C into the executable, it holds the UFL as a
// string in the program and compiles it when the program runs.
//
// This demo illustrates how to:
//
// * Define a variational form in UFL from within a C++ program
// * Compile it at run time and load the generated kernels
// * Build a {cpp:class}`dolfinx::fem::Form` from the loaded
//   `ufcx_form`
//
// ## Equation and problem definition
//
// See {download}`demo_poisson` for the derivation. We solve
//
// \begin{align*}
//    - \nabla \cdot (\kappa \nabla u) &= f \quad {\rm in} \ \Omega, \\
//      u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
//      \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N},
// \end{align*}
//
// on $\Omega = [0, 2] \times [0, 1]$.
//
// ## Implementation
//
// Unlike the ahead-of-time demo, the implementation is contained in a
// single file: there is no separate UFL form file for the build system
// to compile.
//
// Running this demo requires the files:
// {download}`demo_poisson_jit/main.cpp`,
// {download}`demo_poisson_jit/jit.h` and
// {download}`demo_poisson_jit/CMakeLists.txt`.

#include "jit.h"
#include <basix/finite-element.h>
#include <cmath>
#include <cstddef>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/petsc.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <string>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

// The variational problem is written in UFL exactly as it would be in a
// form file, and held as a string literal. The names bound at the top
// level of this source, here `a` and `L`, are the names by which the
// compiled forms are retrieved below.

constexpr std::string_view poisson_ufl = R"(
from basix.ufl import element
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    ds,
    dx,
    grad,
    inner,
)

e = element("Lagrange", "triangle", 1)

coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
g = Coefficient(V)
kappa = Constant(mesh)

a = kappa * inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
)";

// The mesh, element and function space are built exactly as in the
// ahead-of-time demo.

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet, 2);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {32, 16}, mesh::CellType::triangle, part));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::triangle, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(element)));

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);

    // Compile the UFL. This is collective: rank 0 runs FFCx and the C
    // compiler, and the remaining ranks wait and then load the result.
    // The scalar type is taken from `PetscScalar` rather than probed by
    // the build system, so it cannot disagree with the PETSc in use.
    // Compiled forms are cached on disk, keyed by the UFL source and the
    // tools used to build it, so a second run of this program does no
    // work here.
    std::vector<std::string> names = {"a", "L"};
    std::vector<ufcx_form*> forms
        = dolfinx_demo::jit::compile_forms(MPI_COMM_WORLD, poisson_ufl, names,
                                           dolfinx_demo::jit::scalar_type<T>());

    // From here on a JIT-compiled `ufcx_form` is indistinguishable from
    // one linked into the program ahead of time. The element hashes
    // carried by the form are checked against `V` inside
    // {cpp:func}`dolfinx::fem::create_form`, so a mismatch between the
    // UFL and the function space built above is caught here.
    fem::Form<T> a = fem::create_form<T>(*forms[0], {V, V}, {},
                                         {{"kappa", kappa}}, {}, {});
    fem::Form<T> L
        = fem::create_form<T>(*forms[1], {V}, {{"f", f}, {"g", g}}, {}, {}, {});

    // Define boundary condition

    std::vector facets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
              marker[p] = true;
          }
          return marker;
        });
    std::vector bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    fem::DirichletBC<T> bc(0, bdofs, V);

    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy) / 0.02));
          }

          return {f, {f.size()}};
        });

    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(std::sin(5 * x(0, p)));
          return {f, {f.size()}};
        });

    // Assemble and solve, as in the ahead-of-time demo

    auto u = std::make_shared<fem::Function<T>>(V);
    la::petsc::Matrix A(fem::petsc::create_matrix(a), false);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         a, {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                         {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);

    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    lu.set_from_options();

    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    lu.solve(_u.vec(), _b.vec());

    // Update ghost values before output
    u->x()->scatter_fwd();

    // Report the norm of the solution, which must match that of the
    // ahead-of-time demo. la::norm is collective, so every rank must
    // reach it; only rank 0 prints the result.
    auto norm = la::norm(*u->x());
    if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
      spdlog::info("L2 norm of solution vector: {}", norm);

    // Save solution in VTK format. A distinct file name is used so that
    // the output can be compared against the ahead-of-time demo run in
    // the same directory.
    io::VTKFile file(MPI_COMM_WORLD, "u_jit.pvd", "w");
    file.write<T>({*u}, 0);

#ifdef HAS_ADIOS2
    // Save solution in VTX format
    io::VTXWriter<U> vtx(MPI_COMM_WORLD, "u_jit.bp", {u}, "bp4");
    vtx.write(0);
#endif
  }

  PetscFinalize();

  return 0;
}
