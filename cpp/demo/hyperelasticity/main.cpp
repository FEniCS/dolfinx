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
#include <dolfinx/nls/NewtonSolver.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

/// Hyperelastic problem class
class HyperElasticProblem
{
public:
  /// Constructor
  HyperElasticProblem(
      std::shared_ptr<fem::Form<T>> L, std::shared_ptr<fem::Form<T>> J,
      std::vector<std::shared_ptr<const fem::DirichletBC<T>>> bcs)
      : _l(L), _j(J), _bcs(bcs),
        _b(L->function_spaces()[0]->dofmap()->index_map,
           L->function_spaces()[0]->dofmap()->index_map_bs()),
        _matA(la::petsc::Matrix(fem::petsc::create_matrix(*J, "aij"), false))
  {
    auto map = L->function_spaces()[0]->dofmap()->index_map;
    const int bs = L->function_spaces()[0]->dofmap()->index_map_bs();
    std::int32_t size_local = bs * map->size_local();

    std::vector<PetscInt> ghosts(map->ghosts().begin(), map->ghosts().end());
    std::int64_t size_global = bs * map->size_global();
    VecCreateGhostBlockWithArray(map->comm(), bs, size_local, size_global,
                                 ghosts.size(), ghosts.data(),
                                 _b.array().data(), &_b_petsc);
  }

  /// Destructor
  virtual ~HyperElasticProblem()
  {
    if (_b_petsc)
      VecDestroy(&_b_petsc);
  }

  /// @brief  Form
  /// @return
  auto form()
  {
    return [](Vec x)
    {
      VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
      VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
    };
  }

  /// Compute F at current point x
  auto F()
  {
    return [&](const Vec x, Vec)
    {
      // Assemble b and update ghosts
      std::span<T> b(_b.mutable_array());
      std::ranges::fill(b, 0.0);
      fem::assemble_vector<T>(b, *_l);
      VecGhostUpdateBegin(_b_petsc, ADD_VALUES, SCATTER_REVERSE);
      VecGhostUpdateEnd(_b_petsc, ADD_VALUES, SCATTER_REVERSE);

      // Set bcs
      Vec x_local;
      VecGhostGetLocalForm(x, &x_local);
      PetscInt n = 0;
      VecGetSize(x_local, &n);
      const T* array = nullptr;
      VecGetArrayRead(x_local, &array);
      fem::set_bc<T>(b, _bcs, std::span<const T>(array, n), -1.0);
      VecRestoreArrayRead(x, &array);
    };
  }

  /// Compute J = F' at current point x
  auto J()
  {
    return [&](const Vec, Mat A)
    {
      MatZeroEntries(A);
      fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A, ADD_VALUES), *_j,
                           _bcs);
      MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
      fem::set_diagonal(la::petsc::Matrix::set_fn(A, INSERT_VALUES),
                        *_j->function_spaces()[0], _bcs);
      MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    };
  }

  /// RHS vector
  Vec vector() { return _b_petsc; }

  /// Jacobian matrix
  Mat matrix() { return _matA.mat(); }

private:
  std::shared_ptr<fem::Form<T>> _l, _j;
  std::vector<std::shared_ptr<const fem::DirichletBC<T>>> _bcs;
  la::Vector<T> _b;
  Vec _b_petsc = nullptr;
  la::petsc::Matrix _matA;
};

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
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

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element, {3}));

    auto B = std::make_shared<fem::Constant<T>>(std::vector<T>{0, 0, 0});
    auto traction = std::make_shared<fem::Constant<T>>(std::vector<T>{0, 0, 0});

    // Define solution function
    auto u = std::make_shared<fem::Function<T>>(V);
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_hyperelasticity_J_form, {V, V}, {{"u", u}},
                            {{"B", B}, {"T", traction}}, {}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_hyperelasticity_F_form, {V}, {{"u", u}},
                            {{"B", B}, {"T", traction}}, {}, {}));

    auto u_rotation = std::make_shared<fem::Function<T>>(V);
    u_rotation->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          constexpr U scale = 0.005;

          // Center of rotation
          constexpr U x1_c = 0.5;
          constexpr U x2_c = 0.5;

          // Large angle of rotation (60 degrees)
          constexpr U theta = 1.04719755;

          // New coordinates
          std::vector<U> fdata(3 * x.extent(1), 0.0);
          MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
              U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                     std::size_t, 3,
                     MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
              f(fdata.data(), 3, x.extent(1));
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
    std::vector bcs = {
        std::make_shared<const fem::DirichletBC<T>>(std::vector<T>{0, 0, 0},
                                                    bdofs_left, V),
        std::make_shared<const fem::DirichletBC<T>>(u_rotation, bdofs_right)};

    HyperElasticProblem problem(L, a, bcs);
    nls::petsc::NewtonSolver newton_solver(mesh->comm());
    newton_solver.setF(problem.F(), problem.vector());
    newton_solver.setJ(problem.J(), problem.matrix());
    newton_solver.set_form(problem.form());
    newton_solver.rtol = 10 * std::numeric_limits<T>::epsilon();
    newton_solver.atol = 10 * std::numeric_limits<T>::epsilon();

    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    auto [niter, success] = newton_solver.solve(_u.vec());
    std::cout << "Number of Newton iterations: " << niter << std::endl;

    // Compute Cauchy stress. Construct appropriate Basix element for
    // stress.
    constexpr auto family = basix::element::family::P;
    auto cell_type
        = mesh::cell_type_to_basix_type(mesh->topology()->cell_type());
    constexpr int k = 0;
    constexpr bool discontinuous = true;
    basix::FiniteElement S_element = basix::create_element<U>(
        family, cell_type, k, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, discontinuous);
    auto S = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
        mesh, S_element, std::vector<std::size_t>{3, 3}));
    auto sigma_expression = fem::create_expression<T, U>(
        *expression_hyperelasticity_sigma, {{"u", u}}, {});

    auto sigma = fem::Function<T>(S);
    sigma.name = "cauchy_stress";
    sigma.interpolate(sigma_expression);

    // Save solution in VTK format
    io::VTKFile file_u(mesh->comm(), "u.pvd", "w");
    file_u.write<T>({*u}, 0.0);

    // Save Cauchy stress in XDMF format
    io::XDMFFile file_sigma(mesh->comm(), "sigma.xdmf", "w");
    file_sigma.write_mesh(*mesh);
    file_sigma.write_function(sigma, 0.0);
  }

  PetscFinalize();

  return 0;
}
