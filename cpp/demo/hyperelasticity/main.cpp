// # Hyperelasticity
//
// Solve a compressible neo-Hookean model in 3D.

// ## UFL form file
//
// The UFL file is implemented in
// {download}`demo_hyperelasticity/hyperelasticity.py`.
// ````{admonition} UFL form implemented in python
// :class: dropdown
// ![ufl-code]
// ````
//

// ## C++ program

#include "hyperelasticity.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/petsc.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/nls/SNESSolver.h>
#include <format>
#include <functional>
#include <memory>
#include <numbers>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stdexcept>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  dolfinx::common::petsc::check(PetscInitialize(&argc, &argv, nullptr, nullptr),
                                "PetscInitialize");

  // Set the logging thread name to show the process rank
  int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  std::string fmt
      = std::format("[%Y-%m-%d %H:%M:%S.%e] [RANK {}] [%l] %v", mpi_rank);
  spdlog::set_pattern(fmt);
  {
    // Inside the `main` function, we begin by defining a tetrahedral
    // mesh of the domain and the function space on this mesh. Here, we
    // choose to create a unit cube mesh with 25 ( = 24 + 1) vertices in
    // one direction and 17 ( = 16 + 1) vertices in the other two
    // directions. With this mesh, we initialize the (finite element)
    // function space defined by the generated code.

    // Create mesh and define function space
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10},
        mesh::CellType::tetrahedron, graph::partition_graph));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      element, std::vector<std::size_t>{3})));

    auto B = std::make_shared<fem::Constant<T>>(std::vector<T>{0, 0, 0});
    auto traction = std::make_shared<fem::Constant<T>>(std::vector<T>{0, 0, 0});

    // Define solution function
    auto u = std::make_shared<fem::Function<T>>(V);
    fem::Form<T> a
        = fem::create_form<T>(*form_hyperelasticity_J_form, {V, V}, {{"u", u}},
                              {{"B", B}, {"T", traction}}, {}, {});
    fem::Form<T> L
        = fem::create_form<T>(*form_hyperelasticity_F_form, {V}, {{"u", u}},
                              {{"B", B}, {"T", traction}}, {}, {});

    auto u_rotation = std::make_shared<fem::Function<T>>(V);
    u_rotation->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          constexpr U scale = 0.005;

          // Center of rotation
          constexpr U x1_c = 0.5;
          constexpr U x2_c = 0.5;

          // Large angle of rotation (60 degrees)
          constexpr U theta = std::numbers::pi / 3;

          // New coordinates
          std::vector<U> fdata(3 * x.extent(1), 0);
          md::mdspan<U, md::extents<std::size_t, 3, md::dynamic_extent>> f(
              fdata.data(), 3, x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            U x1 = x(1, p);
            U x2 = x(2, p);
            f(1, p) = scale
                      * (x1_c + (x1 - x1_c) * std::cos(theta)
                         - (x2 - x2_c) * std::sin(theta) - x1);
            f(2, p) = scale
                      * (x2_c + (x1 - x1_c) * std::sin(theta)
                         - (x2 - x2_c) * std::cos(theta) - x2);
          }

          return {std::move(fdata), {3, x.extent(1)}};
        });

    // Create Dirichlet boundary conditions
    auto bdofs_left = fem::locate_dofs_geometrical(
        *V,
        [](auto x)
        {
          constexpr U eps = 1.0e-6;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            if (std::abs(x(0, p)) < eps)
              marker[p] = true;
          }
          return marker;
        });
    auto bdofs_right = fem::locate_dofs_geometrical(
        *V,
        [](auto x)
        {
          constexpr U eps = 1.0e-6;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            if (std::abs(x(0, p) - 1) < eps)
              marker[p] = true;
          }
          return marker;
        });
    std::vector bcs
        = {fem::DirichletBC<T>(std::vector<T>{0, 0, 0}, bdofs_left, V),
           fem::DirichletBC<T>(u_rotation, bdofs_right)};

    // `A_layout` and `b_layout` set the layout of the Jacobian and
    // residual that the solver works with. `u_vec` shares data with the
    // degrees-of-freedom of `u`, and holds the initial guess on entry
    // to the solve and the solution on return.
    la::petsc::Matrix A_layout(fem::petsc::create_matrix(a, "aij"), false);
    la::petsc::Vector b_layout(
        la::petsc::create_vector(*V->dofmap()->index_map,
                                 V->dofmap()->index_map_bs()),
        false);
    la::petsc::Vector u_vec(la::petsc::create_vector_wrap(*u->x()), false);
    std::vector<std::reference_wrapper<const fem::DirichletBC<T>>> bcs_ref(
        bcs.begin(), bcs.end());

    // Create the solver, and attach the residual and Jacobian assembly.
    // Each callback assembles at the point `x` into the `b` or `Jmat`
    // it is passed, which may or may not be `b_layout.vec()` or
    // `A_layout.mat()`: a line search, for instance, evaluates the
    // residual in a work vector duplicated from `b_layout.vec()`.
    nls::petsc::SNESSolver solver(mesh->comm());
    solver.set_F([&L, &a, &bcs_ref, &u](const Vec x, Vec b)
                 { fem::petsc::assemble_residual(x, b, L, a, bcs_ref, *u); },
                 b_layout.vec());
    solver.set_J(
        [&a, &bcs_ref, &u](const Vec x, Mat Jmat, Mat)
        { fem::petsc::assemble_jacobian(x, Jmat, nullptr, a, bcs_ref, *u); },
        A_layout.mat());

    // Begin configuring the solver through the PETSc options database.  The
    // Newton update is solved for with a direct LU solver, and a failure to
    // converge raises an error rather than being reported by the return value.
    const U tol = 10 * std::numeric_limits<U>::epsilon();
    common::petsc::set_option("hyperelasticity_ksp_type", "preonly");
    common::petsc::set_option("hyperelasticity_pc_type", "lu");
    common::petsc::set_option("hyperelasticity_snes_rtol", tol);
    common::petsc::set_option("hyperelasticity_snes_atol", tol);
    common::petsc::set_option("hyperelasticity_snes_error_if_not_converged");

    solver.set_options_prefix("hyperelasticity_");
    solver.set_from_options();

    if (solver.solve(u_vec.vec()) < 0)
      throw std::runtime_error("SNES solver did not converge.");
    common::petsc::check(
        VecGhostUpdateBegin(u_vec.vec(), INSERT_VALUES, SCATTER_FORWARD),
        "VecGhostUpdateBegin");
    common::petsc::check(
        VecGhostUpdateEnd(u_vec.vec(), INSERT_VALUES, SCATTER_FORWARD),
        "VecGhostUpdateEnd");

    // The SNES object is available for anything the solver does not
    // wrap, here the number of Newton and linear solver iterations
    PetscInt niter = 0;
    common::petsc::check(SNESGetIterationNumber(solver.snes(), &niter),
                         "SNESGetIterationNumber");
    PetscInt lin_iter = 0;
    common::petsc::check(SNESGetLinearSolveIterations(solver.snes(), &lin_iter),
                         "SNESGetLinearSolveIterations");
    std::cout << "Number of Newton iterations: " << niter << std::endl;
    std::cout << "Number of linear solver iterations: " << lin_iter
              << std::endl;

    // Compute Cauchy stress. Construct appropriate Basix element for
    // stress.
    fem::Expression sigma_expression = fem::create_expression<T, U>(
        *expression_hyperelasticity_sigma, {{"u", u}}, {}, {});

    constexpr auto family = basix::element::family::P;
    auto cell_type
        = mesh::cell_type_to_basix_type(mesh->topology()->cell_type());
    constexpr int k = 0;
    constexpr bool discontinuous = true;
    basix::FiniteElement S_element = basix::create_element<U>(
        family, cell_type, k, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, discontinuous);
    auto S
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      S_element, std::vector<std::size_t>{3, 3})));

    fem::Function<T> sigma(S);
    sigma.name = "cauchy_stress";
    sigma.interpolate(sigma_expression);

    // Save solution in VTK format
    io::VTKFile file_u(mesh->comm(), "u.pvd", "w");
    file_u.write<T>({*u}, 0);

    // Save Cauchy stress in XDMF format
    io::XDMFFile file_sigma(mesh->comm(), "sigma.xdmf", "w");
    file_sigma.write_mesh(*mesh);
    file_sigma.write_function(sigma, 0);
  }

  common::petsc::check(PetscFinalize(), "PetscFinalize");
  return 0;
}
