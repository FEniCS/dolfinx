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
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/nls/NonlinearProblem.h>
#include <format>
#include <memory>
#include <numbers>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

// The residual and Jacobian assembly are handed to
// {cpp:class}`nls::petsc::NonlinearProblem
// <dolfinx::nls::petsc::NonlinearProblem>`, which drives a PETSc SNES.
// The class below owns the data the two callbacks assemble into, and
// the solution function `u` that the forms are evaluated at.

/// Hyperelastic problem class
class HyperElasticProblem
{
public:
  /// Constructor
  HyperElasticProblem(fem::Form<T>& L, fem::Form<T>& J,
                      const std::vector<fem::DirichletBC<T>>& bcs,
                      std::shared_ptr<fem::Function<T>> u)
      : _l(L), _j(J), _bcs(bcs.begin(), bcs.end()), _u_fn(u),
        _b(la::petsc::create_vector(
               *L.function_spaces()[0]->dofmap()->index_map,
               L.function_spaces()[0]->dofmap()->index_map_bs()),
           false),
        _matJ(la::petsc::Matrix(fem::petsc::create_matrix(J, "aij"), false)),
        _u(la::petsc::create_vector_wrap(*u->x()), false),
        _problem(L.function_spaces()[0]->dofmap()->index_map->comm())
  {
    // Attach the assembly callbacks and the data that defines their
    // layout. The callbacks assemble into the vector and matrix they
    // are passed, which are not always those registered here.
    // Capturing `this` is safe as the solver is a member, so the
    // callbacks cannot outlive the problem.
    _problem.set_F(
        [this](const Vec x, Vec b)
        { fem::petsc::assemble_residual(b, x, _l, _j, _bcs, *_u_fn); },
        _b.vec());
    _problem.set_J(
        [this](const Vec x, Mat Jmat, Mat)
        { fem::petsc::assemble_jacobian(Jmat, nullptr, x, _j, _bcs, *_u_fn); },
        _matJ.mat());
    _problem.set_options_prefix("hyperelasticity_");
    _problem.set_from_options();
  }

  /// @brief Solve the nonlinear problem, updating the solution function
  /// `u` that the problem was created with.
  /// @return Number of Newton iterations.
  int solve()
  {
    int iterations = _problem.solve(_u.vec());
    VecGhostUpdateBegin(_u.vec(), INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(_u.vec(), INSERT_VALUES, SCATTER_FORWARD);
    return iterations;
  }

  /// @brief Get the underlying PETSc SNES object, e.g. to set
  /// tolerances or query the convergence reason.
  /// @return The PETSc SNES object.
  SNES snes() const { return _problem.snes(); }

private:
  fem::Form<T>& _l;
  fem::Form<T>& _j;
  std::vector<std::reference_wrapper<const fem::DirichletBC<T>>> _bcs;

  // Solution function that the forms are evaluated at
  std::shared_ptr<fem::Function<T>> _u_fn;

  // Residual vector
  la::petsc::Vector _b;

  // Jacobian matrix
  la::petsc::Matrix _matJ;

  // Solution vector, sharing data with the solution function
  la::petsc::Vector _u;

  // Nonlinear solver
  nls::petsc::NonlinearProblem _problem;
};

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

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
        mesh::CellType::tetrahedron,
        mesh::create_cell_partitioner(mesh::GhostMode::none, 2)));

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

    // Configure the solver through the PETSc options database. The
    // options are read when the problem is created, under the prefix it
    // sets on its SNES object. The Newton update is solved for with a
    // direct LU solver, and a failure to converge raises an error
    // rather than being reported by the return value.
    const U tol = 10 * std::numeric_limits<U>::epsilon();
    la::petsc::options::set("hyperelasticity_ksp_type", "preonly");
    la::petsc::options::set("hyperelasticity_pc_type", "lu");
    la::petsc::options::set("hyperelasticity_snes_rtol", tol);
    la::petsc::options::set("hyperelasticity_snes_atol", tol);
    la::petsc::options::set("hyperelasticity_snes_error_if_not_converged");

    HyperElasticProblem problem(L, a, bcs, u);
    int niter = problem.solve();

    // The SNES object is available for anything the problem does not
    // wrap, here the total number of linear solver iterations
    PetscInt lin_iter = 0;
    SNESGetLinearSolveIterations(problem.snes(), &lin_iter);
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

  PetscFinalize();

  return 0;
}
