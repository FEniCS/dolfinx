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

#include "NewtonSolver.h"
#include "hyperelasticity.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <climits>
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
#include <numbers>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

/// Hyperelastic problem class
class HyperElasticProblem
{
public:
  /// Constructor
  HyperElasticProblem(MPI_Comm comm, fem::Form<T>& L, fem::Form<T>& J,
                      const std::vector<fem::DirichletBC<T>>& bcs)
      : _l(L), _j(J), _bcs(bcs.begin(), bcs.end()),
        _b_vec(L.function_spaces()[0]->dofmap()->index_map,
               L.function_spaces()[0]->dofmap()->index_map_bs()),
        _matA(la::petsc::Matrix(fem::petsc::create_matrix(J, "aij"), false)),
        _solver(comm), _dx(nullptr), _comm(comm)
  {
    auto map = L.function_spaces()[0]->dofmap()->index_map;
    const int bs = L.function_spaces()[0]->dofmap()->index_map_bs();
    std::int32_t size_local = bs * map->size_local();

    std::vector<PetscInt> ghosts(map->ghosts().begin(), map->ghosts().end());
    std::int64_t size_global = bs * map->size_global();
    VecCreateGhostBlockWithArray(map->comm(), bs, size_local, size_global,
                                 ghosts.size(), ghosts.data(),
                                 _b_vec.array().data(), &_b);

    // Create linear solver. Default to LU.
    _solver.set_options_prefix("nls_solve_");
    la::petsc::options::set("nls_solve_ksp_type", "preonly");
    la::petsc::options::set("nls_solve_pc_type", "lu");
    _solver.set_from_options();

    _matJ = _matA.mat();
  }

  /// Destructor
  virtual ~HyperElasticProblem()
  {
    if (_b)
      VecDestroy(&_b);
    if (_dx)
      VecDestroy(&_dx);
  }

  std::pair<int, bool> solve(Vec x)
  {
    // Reset iteration counts
    int iteration = 0;
    int krylov_iterations = 0;
    double residual = -1;
    double residual0 = 0;

    auto converged
        = [&iteration, &residual0, this](const Vec r) -> std::pair<double, bool>
    {
      PetscReal residual = 0;
      VecNorm(r, NORM_2, &residual);

      // Relative residual
      const double relative_residual = residual / residual0;

      // Output iteration number and residual
      if (dolfinx::MPI::rank(_comm.comm()) == 0)
      {
        spdlog::info("Newton iteration {}"
                     ": r (abs) = {} (tol = {}), r (rel) = {} (tol = {})",
                     iteration, residual, atol, relative_residual, rtol);
      }

      // Return true if convergence criterion is met
      if (relative_residual < rtol or residual < atol)
        return {residual, true};
      else
        return {residual, false};
    };

    form(x);
    assert(_b);
    F(x, _b);

    // Check convergence
    bool newton_converged = false;
    std::tie(residual, newton_converged) = converged(_b);

    _solver.set_operators(_matJ, _matJ);

    MatCreateVecs(_matJ, &_dx, nullptr);

    // Start iterations
    while (!newton_converged and iteration < max_it)
    {
      // Compute Jacobian
      assert(_matJ);
      J(x, _matJ);

      // Perform linear solve and update total number of Krylov iterations
      krylov_iterations += _solver.solve(_dx, _b);

      // Update solution
      VecAXPY(x, -relaxation_parameter, _dx);

      // Increment iteration count
      ++iteration;

      // Compute F
      form(x);
      F(x, _b);

      // Initialize _residual0
      if (iteration == 1)
      {
        PetscReal _r = 0;
        VecNorm(_dx, NORM_2, &_r);
        residual0 = _r;
      }

      // Test for convergence
      std::tie(residual, newton_converged) = converged(_b);
    }

    if (newton_converged)
    {
      if (dolfinx::MPI::rank(_comm.comm()) == 0)
      {
        spdlog::info("Newton solver finished in {} iterations and {} linear "
                     "solver iterations.",
                     iteration, krylov_iterations);
      }
    }
    else
    {
      throw std::runtime_error("Newton solver did not converge.");
    }

    return {iteration, newton_converged};
  }

  /// @brief  Form
  /// @return
  void form(Vec x)
  {
    VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
  }

  /// Compute F at current point x
  void F(const Vec x, Vec)
  {
    // Assemble b and update ghosts
    std::span b(_b_vec.array());
    std::ranges::fill(b, 0);
    fem::assemble_vector(b, _l);
    VecGhostUpdateBegin(_b, ADD_VALUES, SCATTER_REVERSE);
    VecGhostUpdateEnd(_b, ADD_VALUES, SCATTER_REVERSE);

    // Set bcs
    Vec x_local;
    VecGhostGetLocalForm(x, &x_local);
    PetscInt n = 0;
    VecGetSize(x_local, &n);
    const T* _x = nullptr;
    VecGetArrayRead(x_local, &_x);
    std::ranges::for_each(_bcs, [b, x = std::span(_x, n)](auto& bc)
                          { bc.get().set(b, x, -1); });
    VecRestoreArrayRead(x_local, &_x);
  }

  /// Compute J = F' at current point x
  void J(const Vec, Mat A)
  {
    MatZeroEntries(A);
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A, ADD_VALUES), _j,
                         _bcs);
    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal(la::petsc::Matrix::set_fn(A, INSERT_VALUES),
                      *_j.function_spaces()[0], _bcs);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  }

  /// @brief Relative convergence tolerance.
  double rtol = 1e-9;

  /// @brief Absolute convergence tolerance.
  double atol = 1e-10;

private:
  fem::Form<T>& _l;
  fem::Form<T>& _j;
  std::vector<std::reference_wrapper<const fem::DirichletBC<T>>> _bcs;
  la::Vector<T> _b_vec;
  Vec _b = nullptr;

  // Jacobian matrix
  la::petsc::Matrix _matA;
  Mat _matJ = nullptr;

  // Linear solver
  dolfinx::la::petsc::KrylovSolver _solver;

  // Solution vector
  Vec _dx = nullptr;

  // MPI communicator
  dolfinx::MPI::Comm _comm;

  /// @brief Maximum number of iterations.
  int max_it = 50;

  /// @brief Relaxation parameter.
  double relaxation_parameter = 1.0;
};

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Set the logging thread name to show the process rank
  int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  std::string fmt = "[%Y-%m-%d %H:%M:%S.%e] [RANK " + std::to_string(mpi_rank)
                    + "] [%l] %v";
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
        mesh::create_cell_partitioner(mesh::GhostMode::none)));

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

    HyperElasticProblem problem(mesh->comm(), L, a, bcs);
    problem.rtol = 10 * std::numeric_limits<T>::epsilon();
    problem.atol = 10 * std::numeric_limits<T>::epsilon();

    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    auto [niter, success] = problem.solve(_u.vec());
    std::cout << "Number of Newton iterations: " << niter << std::endl;

    // Compute Cauchy stress. Construct appropriate Basix element for
    // stress.
    fem::Expression sigma_expression = fem::create_expression<T, U>(
        *expression_hyperelasticity_sigma, {{"u", u}}, {});

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
